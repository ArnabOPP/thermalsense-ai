"""
ThermalSense AI — Physics-Informed Loss Function
Person B — used by pinn_model.py

The physics constraint:
  Urban surface energy balance: Rn = H + LE + G

  Where:
    Rn = net radiation (W/m²)        ← function of albedo, LST, Tatm
    H  = sensible heat flux (W/m²)   ← function of LST, Tatm, wind
    LE = latent heat flux (W/m²)     ← function of NDVI, soil moisture
    G  = ground heat flux (W/m²)     ← function of ISA%, time of day

  We can't measure all fluxes directly, but we can enforce:
    1. LST should INCREASE when albedo decreases (less reflection → more heat)
    2. LST should DECREASE when NDVI increases (evapotranspiration cools)
    3. LST should INCREASE when ISA% increases (more impervious = more heat stored)
    4. LST should be higher than Tatm in urban cores (UHI signature)

  These are implemented as soft penalty terms added to the MSE loss.

Author: Person B
"""

import torch
import torch.nn as nn


class PhysicsLoss(nn.Module):
    """
    Physics-informed loss for urban heat prediction.

    L_total = L_data + lambda_phys * L_physics

    L_data    = MSE(predicted LST, actual LST)
    L_physics = sum of physics constraint violations
    """

    def __init__(self, lambda_phys: float = 0.1):
        """
        Args:
            lambda_phys: Weight of physics loss relative to data loss.
                         Start at 0.1, increase if model ignores physics.
        """
        super().__init__()
        self.lambda_phys = lambda_phys
        self.mse = nn.MSELoss()

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        features: torch.Tensor,
        feature_names: list[str],
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute total loss with physics constraints.

        Args:
            y_pred: Predicted LST (batch,)
            y_true: Actual LST (batch,)
            features: Input feature tensor (batch, n_features)
            feature_names: List of feature names matching columns in features

        Returns:
            (total_loss, loss_components_dict)
        """
        # ── Data loss (MSE) ───────────────────────────────────────────────────
        L_data = self.mse(y_pred, y_true)

        # ── Extract relevant features ─────────────────────────────────────────
        def get_feat(name):
            if name in feature_names:
                idx = feature_names.index(name)
                return features[:, idx]
            return None

        ndvi    = get_feat("ndvi")
        albedo  = get_feat("albedo")
        isa_pct = get_feat("isa_pct")
        tatm    = get_feat("tatm")
        ndwi    = get_feat("ndwi")

        physics_terms = []

        # ── Constraint 1: Higher albedo → lower LST ───────────────────────────
        # Rn decreases with albedo. If albedo is high, LST should be lower.
        # Penalty: when albedo is high but predicted LST is also high.
        if albedo is not None:
            # Normalize albedo to [0,1], expect LST to decrease linearly
            # Soft constraint: d(LST)/d(albedo) should be negative
            # Implemented as: penalize cases where albedo > 0.3 but LST > mean
            high_albedo_mask = (albedo > 0.3).float()
            albedo_violation = high_albedo_mask * torch.relu(y_pred - y_pred.mean())
            physics_terms.append(albedo_violation.mean())

        # ── Constraint 2: Higher NDVI → lower LST (evapotranspiration cooling) ─
        # LE increases with NDVI (vegetation transpires, cools surface).
        # Penalty: when NDVI is high (vegetation) but predicted LST is also high.
        if ndvi is not None:
            high_veg_mask = (ndvi > 0.4).float()
            # Vegetation pixels should have LST below the mean
            ndvi_violation = high_veg_mask * torch.relu(y_pred - y_pred.mean())
            physics_terms.append(ndvi_violation.mean())

        # ── Constraint 3: Higher ISA% → higher LST (impervious surface heating) ─
        # G increases with ISA% (concrete stores more heat than soil/vegetation).
        # Penalty: when ISA is high but predicted LST is below mean.
        if isa_pct is not None:
            high_isa_mask = (isa_pct > 60).float()
            isa_violation = high_isa_mask * torch.relu(y_pred.mean() - y_pred)
            physics_terms.append(isa_violation.mean())

        # ── Constraint 4: LST ≥ Tatm in urban daytime (UHI signature) ─────────
        # Urban surfaces must be warmer than the overlying air during the day.
        # This is the defining characteristic of the Urban Heat Island.
        # Penalty: when predicted LST < atmospheric temperature.
        if tatm is not None:
            uhi_violation = torch.relu(tatm - y_pred)  # positive when LST < Tatm
            physics_terms.append(uhi_violation.mean())

        # ── Constraint 5: Water bodies are cooler ────────────────────────────
        # NDWI > 0.2 indicates water. Water should have lower LST than land.
        if ndwi is not None:
            water_mask = (ndwi > 0.2).float()
            land_mean_lst = (y_pred * (1 - water_mask)).sum() / ((1 - water_mask).sum() + 1e-8)
            water_violation = water_mask * torch.relu(y_pred - land_mean_lst + 5.0)
            physics_terms.append(water_violation.mean())

        # ── Combine physics terms ─────────────────────────────────────────────
        if physics_terms:
            L_physics = torch.stack(physics_terms).mean()
        else:
            L_physics = torch.tensor(0.0, device=y_pred.device)

        L_total = L_data + self.lambda_phys * L_physics

        return L_total, {
            "L_total":   float(L_total.item()),
            "L_data":    float(L_data.item()),
            "L_physics": float(L_physics.item()),
            "n_constraints": len(physics_terms),
        }

    def physics_violation_rate(
        self,
        y_pred: torch.Tensor,
        features: torch.Tensor,
        feature_names: list[str],
    ) -> float:
        """
        Compute what % of predictions violate at least one physics constraint.
        Used for model card reporting.
        """
        violations = torch.zeros(len(y_pred), device=y_pred.device)

        def get_feat(name):
            if name in feature_names:
                return features[:, feature_names.index(name)]
            return None

        ndvi   = get_feat("ndvi")
        tatm   = get_feat("tatm")
        isa    = get_feat("isa_pct")

        if ndvi is not None:
            violations += ((ndvi > 0.4) & (y_pred > y_pred.mean())).float()
        if tatm is not None:
            violations += (y_pred < tatm).float()
        if isa is not None:
            violations += ((isa > 60) & (y_pred < y_pred.mean())).float()

        return float((violations > 0).float().mean().item())
