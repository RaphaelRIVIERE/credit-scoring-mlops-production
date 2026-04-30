from typing import Optional
from pydantic import BaseModel, Field


class ClientFeatures(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "DAYS_BIRTH": -12000,
                    "DAYS_EMPLOYED": -2000,
                    "AMT_INCOME_TOTAL": 180000.0,
                    "AMT_CREDIT": 450000.0,
                    "AMT_ANNUITY": 22500.0,
                    "CODE_GENDER": "M",
                    "CNT_CHILDREN": 1,
                    "CNT_FAM_MEMBERS": 3.0,
                    "NAME_CONTRACT_TYPE": "Cash loans",
                    "NAME_FAMILY_STATUS": "Married",
                    "NAME_HOUSING_TYPE": "House / apartment",
                    "NAME_INCOME_TYPE": "Working",
                    "NAME_EDUCATION_TYPE": "Secondary / secondary special",
                    "NAME_TYPE_SUITE": "Unaccompanied",
                    "OCCUPATION_TYPE": "Laborers",
                    "ORGANIZATION_TYPE": "Business Entity Type 3",
                    "FLAG_OWN_CAR": "N",
                    "FLAG_OWN_REALTY": "Y",
                    "AMT_GOODS_PRICE": 400000.0,
                    "DAYS_REGISTRATION": -3000.0,
                    "DAYS_ID_PUBLISH": -2000.0,
                    "DAYS_LAST_PHONE_CHANGE": -500.0,
                    "DAYS_EMPLOYED_ANOM": False,
                    "HOUR_APPR_PROCESS_START": 10,
                    "WEEKDAY_APPR_PROCESS_START": "TUESDAY",
                    "EXT_SOURCE_1": 0.5,
                    "EXT_SOURCE_2": 0.6,
                    "EXT_SOURCE_3": 0.55,
                    "REGION_POPULATION_RELATIVE": 0.025,
                    "REGION_RATING_CLIENT": 2,
                    "REGION_RATING_CLIENT_W_CITY": 2,
                    "REG_REGION_NOT_LIVE_REGION": 0,
                    "REG_REGION_NOT_WORK_REGION": 0,
                    "REG_CITY_NOT_LIVE_CITY": 0,
                    "REG_CITY_NOT_WORK_CITY": 0,
                    "LIVE_REGION_NOT_WORK_REGION": 0,
                    "LIVE_CITY_NOT_WORK_CITY": 0,
                    "FLOORSMAX_AVG": 0.17,
                    "FLOORSMAX_MODE": 0.17,
                    "FLOORSMAX_MEDI": 0.17,
                    "TOTALAREA_MODE": 0.05,
                    "YEARS_BEGINEXPLUATATION_AVG": 0.98,
                    "YEARS_BEGINEXPLUATATION_MODE": 0.98,
                    "YEARS_BEGINEXPLUATATION_MEDI": 0.98,
                    "EMERGENCYSTATE_MODE": "No",
                    "FLAG_MOBIL": 1,
                    "FLAG_EMP_PHONE": 1,
                    "FLAG_WORK_PHONE": 0,
                    "FLAG_CONT_MOBILE": 1,
                    "FLAG_PHONE": 0,
                    "FLAG_EMAIL": 0,
                    "FLAG_DOCUMENT_2": 0,
                    "FLAG_DOCUMENT_3": 1,
                    "FLAG_DOCUMENT_4": 0,
                    "FLAG_DOCUMENT_5": 0,
                    "FLAG_DOCUMENT_6": 0,
                    "FLAG_DOCUMENT_7": 0,
                    "FLAG_DOCUMENT_8": 0,
                    "FLAG_DOCUMENT_9": 0,
                    "FLAG_DOCUMENT_10": 0,
                    "FLAG_DOCUMENT_11": 0,
                    "FLAG_DOCUMENT_12": 0,
                    "FLAG_DOCUMENT_13": 0,
                    "FLAG_DOCUMENT_14": 0,
                    "FLAG_DOCUMENT_15": 0,
                    "FLAG_DOCUMENT_16": 0,
                    "FLAG_DOCUMENT_17": 0,
                    "FLAG_DOCUMENT_18": 0,
                    "FLAG_DOCUMENT_19": 0,
                    "FLAG_DOCUMENT_20": 0,
                    "FLAG_DOCUMENT_21": 0,
                    "OBS_30_CNT_SOCIAL_CIRCLE": 2.0,
                    "DEF_30_CNT_SOCIAL_CIRCLE": 0.0,
                    "OBS_60_CNT_SOCIAL_CIRCLE": 2.0,
                    "DEF_60_CNT_SOCIAL_CIRCLE": 0.0,
                    "AMT_REQ_CREDIT_BUREAU_HOUR": 0.0,
                    "AMT_REQ_CREDIT_BUREAU_DAY": 0.0,
                    "AMT_REQ_CREDIT_BUREAU_WEEK": 0.0,
                    "AMT_REQ_CREDIT_BUREAU_MON": 1.0,
                    "AMT_REQ_CREDIT_BUREAU_QRT": 1.0,
                    "AMT_REQ_CREDIT_BUREAU_YEAR": 3.0,
                    "bureau_count": 5.0,
                    "bureau_active_count": 2.0,
                    "actif_count": 2.0,
                    "cloture_count": 3.0,
                    "bureau_overdue_mean": 0.0,
                    "bureau_debt_mean": 50000.0,
                    "actif_debt_mean": 60000.0,
                    "cloture_debt_mean": 30000.0,
                    "prev_count": 3.0,
                    "approve_count": 2.0,
                    "refuse_count": 1.0,
                    "prev_refused_count": 1.0,
                    "taux_refus": 0.33,
                    "prev_credit_mean": 200000.0,
                    "approve_credit_mean": 250000.0,
                    "refuse_credit_mean": 150000.0,
                    "a_carte_credit": 0.0,
                    "cc_balance_mean": 0.0,
                    "cc_dpd_mean": 0.0,
                    "cc_utilisation_mean": 0.0,
                    "pos_dpd_mean": 0.0,
                    "pos_dpd_max": 0.0,
                    "inst_retard_mean": 0.0,
                    "inst_retard_max": 0.0,
                    "inst_diff_mean": -500.0,
                    "a_eu_retard": 0.0,
                }
            ]
        }
    }

    # Requis par feature_engineering
    DAYS_BIRTH: float = Field(
        ..., lt=0, description="Âge en jours (valeur négative, ex: -12000 ≈ 33 ans)."
    )
    DAYS_EMPLOYED: float = Field(
        ...,
        description=(
            "Ancienneté professionnelle en jours (valeur négative). "
            "La valeur 365243 indique que le client est sans emploi."
        ),
    )
    AMT_INCOME_TOTAL: float = Field(
        ..., gt=0, description="Revenu annuel du client (en devise locale)."
    )
    AMT_CREDIT: float = Field(
        ..., gt=0, description="Montant total du crédit demandé."
    )
    AMT_ANNUITY: float = Field(
        ..., gt=0, description="Mensualité du crédit."
    )

    # Optionnels
    CODE_GENDER: Optional[str] = None
    CNT_CHILDREN: Optional[int] = None
    CNT_FAM_MEMBERS: Optional[float] = None
    NAME_CONTRACT_TYPE: Optional[str] = None
    NAME_FAMILY_STATUS: Optional[str] = None
    NAME_HOUSING_TYPE: Optional[str] = None
    NAME_INCOME_TYPE: Optional[str] = None
    NAME_EDUCATION_TYPE: Optional[str] = None
    NAME_TYPE_SUITE: Optional[str] = None
    OCCUPATION_TYPE: Optional[str] = None
    ORGANIZATION_TYPE: Optional[str] = None
    FLAG_OWN_CAR: Optional[str] = None
    FLAG_OWN_REALTY: Optional[str] = None
    AMT_GOODS_PRICE: Optional[float] = None
    DAYS_REGISTRATION: Optional[float] = None
    DAYS_ID_PUBLISH: Optional[float] = None
    DAYS_LAST_PHONE_CHANGE: Optional[float] = None
    DAYS_EMPLOYED_ANOM: Optional[bool] = None
    HOUR_APPR_PROCESS_START: Optional[int] = None
    WEEKDAY_APPR_PROCESS_START: Optional[str] = None
    EXT_SOURCE_1: Optional[float] = None
    EXT_SOURCE_2: Optional[float] = None
    EXT_SOURCE_3: Optional[float] = None
    REGION_POPULATION_RELATIVE: Optional[float] = None
    REGION_RATING_CLIENT: Optional[int] = None
    REGION_RATING_CLIENT_W_CITY: Optional[int] = None
    REG_REGION_NOT_LIVE_REGION: Optional[int] = None
    REG_REGION_NOT_WORK_REGION: Optional[int] = None
    REG_CITY_NOT_LIVE_CITY: Optional[int] = None
    REG_CITY_NOT_WORK_CITY: Optional[int] = None
    LIVE_REGION_NOT_WORK_REGION: Optional[int] = None
    LIVE_CITY_NOT_WORK_CITY: Optional[int] = None
    FLOORSMAX_AVG: Optional[float] = None
    FLOORSMAX_MODE: Optional[float] = None
    FLOORSMAX_MEDI: Optional[float] = None
    TOTALAREA_MODE: Optional[float] = None
    YEARS_BEGINEXPLUATATION_AVG: Optional[float] = None
    YEARS_BEGINEXPLUATATION_MODE: Optional[float] = None
    YEARS_BEGINEXPLUATATION_MEDI: Optional[float] = None
    EMERGENCYSTATE_MODE: Optional[str] = None
    FLAG_MOBIL: Optional[int] = None
    FLAG_EMP_PHONE: Optional[int] = None
    FLAG_WORK_PHONE: Optional[int] = None
    FLAG_CONT_MOBILE: Optional[int] = None
    FLAG_PHONE: Optional[int] = None
    FLAG_EMAIL: Optional[int] = None
    FLAG_DOCUMENT_2: Optional[int] = None
    FLAG_DOCUMENT_3: Optional[int] = None
    FLAG_DOCUMENT_4: Optional[int] = None
    FLAG_DOCUMENT_5: Optional[int] = None
    FLAG_DOCUMENT_6: Optional[int] = None
    FLAG_DOCUMENT_7: Optional[int] = None
    FLAG_DOCUMENT_8: Optional[int] = None
    FLAG_DOCUMENT_9: Optional[int] = None
    FLAG_DOCUMENT_10: Optional[int] = None
    FLAG_DOCUMENT_11: Optional[int] = None
    FLAG_DOCUMENT_12: Optional[int] = None
    FLAG_DOCUMENT_13: Optional[int] = None
    FLAG_DOCUMENT_14: Optional[int] = None
    FLAG_DOCUMENT_15: Optional[int] = None
    FLAG_DOCUMENT_16: Optional[int] = None
    FLAG_DOCUMENT_17: Optional[int] = None
    FLAG_DOCUMENT_18: Optional[int] = None
    FLAG_DOCUMENT_19: Optional[int] = None
    FLAG_DOCUMENT_20: Optional[int] = None
    FLAG_DOCUMENT_21: Optional[int] = None
    OBS_30_CNT_SOCIAL_CIRCLE: Optional[float] = None
    DEF_30_CNT_SOCIAL_CIRCLE: Optional[float] = None
    OBS_60_CNT_SOCIAL_CIRCLE: Optional[float] = None
    DEF_60_CNT_SOCIAL_CIRCLE: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_HOUR: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_DAY: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_WEEK: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_MON: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_QRT: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_YEAR: Optional[float] = None
    bureau_count: Optional[float] = None
    bureau_active_count: Optional[float] = None
    actif_count: Optional[float] = None
    cloture_count: Optional[float] = None
    bureau_overdue_mean: Optional[float] = None
    bureau_debt_mean: Optional[float] = None
    actif_debt_mean: Optional[float] = None
    cloture_debt_mean: Optional[float] = None
    prev_count: Optional[float] = None
    approve_count: Optional[float] = None
    refuse_count: Optional[float] = None
    prev_refused_count: Optional[float] = None
    taux_refus: Optional[float] = None
    prev_credit_mean: Optional[float] = None
    approve_credit_mean: Optional[float] = None
    refuse_credit_mean: Optional[float] = None
    a_carte_credit: Optional[float] = None
    cc_balance_mean: Optional[float] = None
    cc_dpd_mean: Optional[float] = None
    cc_utilisation_mean: Optional[float] = None
    pos_dpd_mean: Optional[float] = None
    pos_dpd_max: Optional[float] = None
    inst_retard_mean: Optional[float] = None
    inst_retard_max: Optional[float] = None
    inst_diff_mean: Optional[float] = None
    a_eu_retard: Optional[float] = None


class PredictionResponse(BaseModel):
    score: float = Field(..., ge=0, le=1, description="Probabilité de défaut (0 = bon payeur, 1 = défaut)")
    decision: str = Field(..., description="'approved' ou 'rejected'")
