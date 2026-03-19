# Changelog

## [2.0.0] - 2025

### Added
- JWT authentication (register, login, /api/auth/me)
- Model file upload endpoint (/api/models/upload)
- Audit run trigger endpoint (/api/audits/{id}/run)
- JSON export endpoint (/api/audits/{id}/export)
- Full dark-theme web dashboard (Bootstrap 5 + Chart.js)
- Radar chart visualization on audit detail page
- Docker + docker-compose support
- GitHub Actions CI pipeline
- .env support via python-dotenv

### Fixed
- Missing is_admin field on User model
- Missing web/routes.py blueprint
- model_metadata JSON column rename conflict
- Session management in audit service

### Changed
- Upgraded to Flask 3.0, SQLAlchemy 2.0
- All write endpoints now require JWT
- Audit service auto-generates sample data if none found

## [1.0.0] - 2025
- Initial prototype release
