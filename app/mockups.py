from __future__ import annotations

from flask import Blueprint, render_template

mockups = Blueprint("mockups", __name__, url_prefix="/mockups")


@mockups.get("/final-dashboard/")
def final_dashboard():
    return render_template("mockups/final_dashboard.html")
