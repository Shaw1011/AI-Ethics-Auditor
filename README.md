# AI Ethics Auditor

![CI](https://github.com/Shaw1011/AI-Ethics-Auditor/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue?logo=python)
![Flask](https://img.shields.io/badge/flask-3.1-green?logo=flask)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Security](https://img.shields.io/badge/security-bandit-yellow)

A production-grade system for auditing AI/ML models against ethical standards — covering **fairness**, **explainability**, and **robustness** — with JWT authentication, a dark-theme web dashboard, paginated REST API, and Docker support.

---

## Features

| Category | Details |
|---|---|
| **Fairness** | Demographic Parity, Equalized Odds, Disparate Impact Ratio, Statistical Parity via [fairlearn](https://fairlearn.org) |
| **Explainability** | SHAP-based feature importance + scikit-learn fallback, model complexity score |
| **Robustness** | Prediction stability under Gaussian noise (20 perturbations, seeded) |
| **Security** | JWT (access + refresh tokens), token blocklist (logout invalidation), bcrypt pw hashing, rate limiting, CORS, security headers (Talisman), input validation (marshmallow), SHA-256 model file verification |
| **API** | Full CRUD, pagination, structured error envelopes, JSON export |
| **Upload** | `.joblib` only — no pickle; hash verified on every load |
| **Infra** | Docker multi-stage build, non-root container user, GitHub Actions CI |

---

## Quickstart

```bash
git clone https://github.com/Shaw1011/AI-Ethics-Auditor.git
cd AI-Ethics-Auditor
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # edit SECRET_KEY and JWT_SECRET_KEY
python main.py
```

Open http://127.0.0.1:5000

---

## Docker

```bash
cp .env.example .env   # fill in secrets
docker-compose up --build
```

---

## API Reference

All write endpoints require `Authorization: Bearer <token>`.

| Method | Endpoint | Auth | Description |
|--------|----------|:----:|-------------|
| `POST` | `/api/auth/register` | — | Create account |
| `POST` | `/api/auth/login` | — | Get access + refresh tokens |
| `POST` | `/api/auth/refresh` | Refresh | Renew access token |
| `DELETE` | `/api/auth/logout` | JWT | Revoke token |
| `GET` | `/api/auth/me` | JWT | Current user |
| `GET` | `/api/models/?page=1&size=20` | — | List models (paginated) |
| `POST` | `/api/models/` | JWT | Register model |
| `POST` | `/api/models/upload` | JWT | Upload `.joblib` file |
| `PATCH` | `/api/models/{id}` | JWT | Update model |
| `DELETE` | `/api/models/{id}` | JWT | Delete model |
| `GET` | `/api/audits/?status=completed` | — | List audits (paginated, filterable) |
| `POST` | `/api/audits/` | JWT | Create audit |
| `POST` | `/api/audits/{id}/run` | JWT | Execute audit |
| `GET` | `/api/audits/{id}/export?format=json` | — | Export report |
| `DELETE` | `/api/audits/{id}` | JWT | Delete audit |

### Error Envelope

All errors follow a consistent structure:
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Validation failed.",
    "details": { "username": ["Length must be between 3 and 50."] }
  }
}
```

---

## Running Audits

1. **Prepare your model** — save with `joblib.dump(model, "model.joblib")`
2. **Prepare your data** — a CSV with `target` and `protected_attribute` columns at `data/testing_data.csv`
3. **Upload the model** — `POST /api/models/upload`
4. **Create an audit** — `POST /api/audits/`
5. **Run the audit** — `POST /api/audits/{id}/run`
6. **View results** — `GET /api/audits/{id}` or open the dashboard

---

## Metrics Explained

| Metric | Category | Ideal | Concern Threshold |
|--------|----------|-------|-------------------|
| Demographic Parity Difference | Fairness | 0 | > 0.10 |
| Equalized Odds Difference | Fairness | 0 | > 0.10 |
| Disparate Impact Ratio | Fairness | 1.0 | < 0.80 (80% rule) |
| Statistical Parity | Fairness | 0 | > 0.10 |
| Feature Importance | Explainability | Higher = more transparent | — |
| Model Complexity | Explainability | Lower = more explainable | — |
| Prediction Stability | Robustness | 1.0 | < 0.85 |

---

## Security Design

- **No pickle** — models are stored and loaded as `.joblib` only
- **Hash verification** — SHA-256 of every uploaded file is stored and re-verified on each load
- **Token blocklist** — logout actually invalidates tokens (stored in DB)
- **bcrypt** — passwords hashed with work factor 12
- **Rate limiting** — login: 20/5min, register: 10/hr
- **Input validation** — every endpoint validated with marshmallow schemas before DB access
- **No stack traces in production** — generic 500 responses, full trace only in server logs
- **Security headers** — CSP, HSTS, X-Frame-Options: DENY, X-Content-Type-Options (via Talisman)

---

## Testing

```bash
pytest tests/ -v --cov=app --cov-report=term-missing
```

Test coverage includes: auth flows, CRUD, pagination, security edge cases (SQL injection strings, oversized inputs, tampered tokens, wrong file types).

---

## Project Structure

```
AI-Ethics-Auditor/
├── app/
│   ├── api/            # auth, models, audits, users
│   ├── middleware/     # global error handlers
│   ├── models/         # SQLAlchemy ORM (user, model, audit, token_blocklist)
│   ├── services/       # audit engine, model loader, report generator
│   ├── static/         # CSS, JS
│   ├── templates/      # Jinja2 HTML (dashboard, models, audits, login)
│   ├── utils/          # errors, pagination, security, validators
│   └── web/            # page routes
├── tests/              # pytest suite (auth, models, audits, security)
├── .github/workflows/  # CI: test + bandit + safety + docker build
├── pyproject.toml
├── pytest.ini
├── Dockerfile          # multi-stage, non-root
├── docker-compose.yml
└── main.py
```

---

## Author

**Edge Shaw** · [GitHub](https://github.com/Shaw1011)

## License

MIT
