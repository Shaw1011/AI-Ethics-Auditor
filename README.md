# AI Ethics Auditor

A system for auditing AI models for ethical considerations including fairness, explainability, and robustness.

## Features

- Audit AI models for various ethical metrics
- Track model performance across multiple fairness dimensions
- Generate comprehensive ethics reports
- API-first design for integration with existing ML pipelines

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-ethics-auditor.git
cd ai-ethics-auditor
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python main.py
```

## Usage

The application provides a REST API for managing models and audits:

- `/api/models` - Manage AI models
- `/api/audits` - Manage and run ethics audits
- `/api/users` - Manage users

## Development

### Running Tests

```bash
pytest
```

### Project Structure

- `app/` - Application code
  - `api/` - API routes
  - `models/` - Database models
  - `services/` - Business logic
- `tests/` - Test cases
- `data/` - Data storage (created on first run)

## License

MIT 