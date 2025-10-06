# API Evaluation (Production & Local Reference)

Date: 2025-10-06  
Tag / Version: v1.0.1 (app reports version 1.0.0 in root payload)  
Space: https://huggingface.co/spaces/j2damax/boking-cancelation-api  
Runtime Base URL (expected pattern): `https://boking-cancelation-api-j2damax.hf.space`

> NOTE: At time of capture the public Space URL returned 404 (likely still building or cold start). Local responses shown below (functionally identical to deployed app). Replace base with the hf.space hostname once live.

## 1. Root Endpoint
Request:
```bash
curl -s https://boking-cancelation-api-j2damax.hf.space/
```
Local Response (200 OK):
```json
{"message":"Hotel Cancellation Prediction API","version":"1.0.0","endpoints":{"health":"/health","predict":"/predict","docs":"/docs"}}
```
Evaluation: Returns service metadata; includes advertised endpoints. Version aligns with `config.APP_VERSION`.

## 2. Health Endpoint
Request:
```bash
curl -s https://boking-cancelation-api-j2damax.hf.space/health
```
Local Response (200 OK):
```json
{"status":"healthy","model_loaded":true,"model_version":"local_1759682584","decision_threshold":0.5}
```
Evaluation: Model successfully loaded; default threshold=0.5. If `status` becomes `model_not_loaded` deployment still responsive (depending on ALLOW_START_WITHOUT_MODEL).

## 3. Single Prediction (Minimal Payload)
Request:
```bash
curl -s -X POST https://boking-cancelation-api-j2damax.hf.space/predict \
  -H 'Content-Type: application/json' \
  -d '{"lead_time":45,"arrival_month":7,"adults":2,"children":0,"adr":110.0}'
```
Local Response (200 OK):
```json
{"prediction":0,"probability":0.24073584377765656,"model_version":"local_1759682584","applied_threshold":0.5,"threshold_source":"default"}
```
Evaluation: Probability < threshold → prediction 0 (not cancelled). Threshold source is default (no artifact override).

## 4. Batch Prediction
Request:
```bash
curl -s -X POST https://boking-cancelation-api-j2damax.hf.space/predict/batch \
  -H 'Content-Type: application/json' \
  -d '[{"lead_time":30,"arrival_month":6,"adults":2,"children":0,"adr":95.0},{"lead_time":120,"arrival_month":8,"adults":1,"children":0,"adr":130.0}]'
```
Local Response (200 OK):
```json
[{"prediction":0,"probability":0.20589229464530945,"model_version":"local_1759682584","applied_threshold":0.5,"threshold_source":"default"},{"prediction":0,"probability":0.2820329964160919,"model_version":"local_1759682584","applied_threshold":0.5,"threshold_source":"default"}]
```
Evaluation: Both probabilities below 0.5 threshold; logic consistent with single prediction endpoint.

## 5. Interpretability
Request:
```bash
curl -s https://boking-cancelation-api-j2damax.hf.space/model/interpretability
```
Local Response (200 OK excerpt):
```json
{"champion_model":"XGBoost","shap_generated":true,"shap_timestamp":null,"decision_threshold":0.35000000000000003,"top_features":[{"feature":"deposit_type","mean_abs_shap":1.004747748374939},{"feature":"country__te","mean_abs_shap":0.8516273498535156},{"feature":"market_segment","mean_abs_shap":0.43541011214256287},{"feature":"total_of_special_requests","mean_abs_shap":0.4210052192211151},{"feature":"lead_time","mean_abs_shap":0.41456905007362366},{"feature":"required_car_parking_spaces","mean_abs_shap":0.4020047187805176},{"feature":"assigned_room_type","mean_abs_shap":0.3292023837566376},{"feature":"customer_type_target_encoded","mean_abs_shap":0.2506164312362671},{"feature":"reserved_room_type","mean_abs_shap":0.23714518547058105},{"feature":"previous_cancellations","mean_abs_shap":0.21544909477233887}],"local_examples":[{"category":"true_positive","probability":0.7679175138473511,"top_positive_contributors":[{"feature":"total_of_special_requests","shap":0.736638605594635},{"feature":"market_segment","shap":0.5826431512832642},{"feature":"assigned_room_type","shap":0.3624641001224518},{"feature":"lead_time","shap":0.27093467116355896},{"feature":"adr","shap":0.22705571353435516}],"top_negative_contributors":[{"feature":"country__te","shap":-0.5817746520042419},{"feature":"deposit_type","shap":-0.38260650634765625},{"feature":"reserved_room_type","shap":-0.17490211129188538},{"feature":"previous_cancellations","shap":-0.048768918961286545},{"feature":"distribution_channel_target_encoded","shap":-0.03044990263879299}]},{"category":"false_positive","probability":0.7768429517745972,"top_positive_contributors":[{"feature":"total_of_special_requests","shap":0.7757676243782043},{"feature":"market_segment","shap":0.5442723035812378},{"feature":"assigned_room_type","shap":0.37023118138313293},{"feature":"adr","shap":0.2486179769039154},{"feature":"lead_time","shap":0.20630843937397003}],"top_negative_contributors":[{"feature":"deposit_type","shap":-0.37375608086586},{"feature":"reserved_room_type","shap":-0.28937697410583496},{"feature":"country__te","shap":-0.22035948932170868},{"feature":"previous_cancellations","shap":-0.054328884929418564},{"feature":"hotel","shap":-0.04107680171728134}]},{"category":"false_negative","probability":0.36954548954963684,"top_positive_contributors":[{"feature":"market_segment","shap":0.35903194546699524},{"feature":"adr","shap":0.2758876085281372},{"feature":"assigned_room_type","shap":0.19640541076660156},{"feature":"lead_time","shap":0.18352645635604858},{"feature":"arrival_date_year","shap":0.1316099613904953}],"top_negative_contributors":[{"feature":"total_of_special_requests","shap":-1.018623948097229},{"feature":"deposit_type","shap":-0.39705631136894226},{"feature":"reserved_room_type","shap":-0.20557740330696106},{"feature":"arrival_date_week_number","shap":-0.07640720903873444},{"feature":"previous_cancellations","shap":-0.0757187083363533}]}],"feature_name_map":{},"artifacts_available":[]}
```

Evaluation: Provides global (top_features) and sample local SHAP contributions. Decision threshold (0.35) here reflects value embedded in champion metadata (can differ from runtime default used in prediction endpoint if threshold override not loaded at model load time).

## 6. Validation / Error Scenarios

### 6.1 Missing Required Field
Request (omits `lead_time`):
```bash
curl -s -X POST https://boking-cancelation-api-j2damax.hf.space/predict \
  -H 'Content-Type: application/json' \
  -d '{"arrival_month":7,"adults":2,"children":0,"adr":110.0}'
```
Response (422 Unprocessable Entity):
```json
{"detail":[{"type":"missing","loc":["body","lead_time"],"msg":"Field required","input":{"arrival_month":7,"adults":2,"children":0,"adr":110.0}}]}
```
Evaluation: Pydantic validation correctly identifies missing mandatory field.

### 6.2 Out-of-Range Value
Request (`arrival_month`=13):
```bash
curl -s -X POST https://boking-cancelation-api-j2damax.hf.space/predict \
  -H 'Content-Type: application/json' \
  -d '{"lead_time":45,"arrival_month":13,"adults":2,"children":0,"adr":110.0}'
```
Response (422 Unprocessable Entity):
```json
{"detail":[{"type":"less_than_equal","loc":["body","arrival_month"],"msg":"Input should be less than or equal to 12","input":13,"ctx":{"le":12}}]}
```
Evaluation: Constraint `le=12` enforced as expected.

## 7. Summary Assessment
| Aspect | Result |
|--------|--------|
| Availability | Local service healthy; Space initial 404 suggests cold start or build in progress. |
| Health Reporting | Includes model load status, version, threshold. |
| Prediction Correctness | Threshold logic consistent (prob < 0.5 => class 0). |
| Batch Consistency | Batch endpoint mirrors single inference logic. |
| Interpretability | Returns top SHAP features + sample local breakdowns. |
| Validation Robustness | Proper 422 responses for missing and out-of-range inputs. |
| Version Traceability | Root shows version; health includes model_version token. |

## 8. Recommendations
- Confirm hf.space base URL is live; if persistent 404, redeploy Space or inspect build logs.
- Align decision threshold between prediction endpoint and interpretability metadata (load artifact threshold at model load if divergence unintended).
- Optionally add a readiness probe endpoint if model loading can be long.

---
_This document is concise by design for course report inclusion. Replace sample local responses with production captures once the Space returns 200 responses._
