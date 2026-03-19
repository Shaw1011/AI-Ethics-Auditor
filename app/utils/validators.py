"""
Request validation schemas using marshmallow.
All user input is validated here before touching the DB.
"""
from marshmallow import Schema, fields, validate, validates, ValidationError, post_load


class RegisterSchema(Schema):
    username = fields.Str(
        required=True,
        validate=[validate.Length(min=3, max=50), validate.Regexp(r"^[a-zA-Z0-9_.-]+$")],
    )
    email = fields.Email(required=True, validate=validate.Length(max=254))
    password = fields.Str(
        required=True,
        validate=validate.Length(min=8, max=72),
        load_only=True,
    )
    is_admin = fields.Bool(load_default=False)


class LoginSchema(Schema):
    username = fields.Str(required=True, validate=validate.Length(min=1, max=50))
    password = fields.Str(required=True, load_only=True)


class ModelCreateSchema(Schema):
    name = fields.Str(required=True, validate=validate.Length(min=1, max=100))
    description = fields.Str(validate=validate.Length(max=1000), load_default="")
    model_type = fields.Str(
        required=True,
        validate=validate.OneOf(["classification", "regression", "nlp", "vision", "other"]),
    )
    version = fields.Str(
        validate=[validate.Length(max=20), validate.Regexp(r"^[\w.\-]+$")],
        load_default="1.0",
    )


class ModelUpdateSchema(Schema):
    name = fields.Str(validate=validate.Length(min=1, max=100))
    description = fields.Str(validate=validate.Length(max=1000))
    version = fields.Str(
        validate=[validate.Length(max=20), validate.Regexp(r"^[\w.\-]+$")]
    )


class AuditCreateSchema(Schema):
    name = fields.Str(required=True, validate=validate.Length(min=1, max=100))
    description = fields.Str(validate=validate.Length(max=1000), load_default="")
    model_id = fields.Int(required=True, validate=validate.Range(min=1))


# Shared validation helper
def load_or_422(schema: Schema, data: dict) -> tuple[dict | None, object | None]:
    """
    Attempt to deserialize data with schema.
    Returns (data, None) on success or (None, error_response) on failure.
    """
    from app.utils.errors import validation_error
    try:
        return schema.load(data), None
    except ValidationError as exc:
        return None, validation_error(exc.messages)
