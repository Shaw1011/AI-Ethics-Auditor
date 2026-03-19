"""Web page routes (server-rendered via Jinja2)."""
from flask import Blueprint, render_template

web_bp = Blueprint("web", __name__)


@web_bp.route("/")
def index():
    return render_template("index.html")


@web_bp.route("/models")
def models_page():
    return render_template("models.html")


@web_bp.route("/audits")
def audits_page():
    return render_template("audits.html")


@web_bp.route("/audits/<int:audit_id>")
def audit_detail(audit_id: int):
    return render_template("audit_detail.html", audit_id=audit_id)


@web_bp.route("/login")
def login_page():
    return render_template("login.html")
