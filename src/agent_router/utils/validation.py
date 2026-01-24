"""Input validation utilities."""

from typing import List
from agent_router.core.types import JsonObject
from pydantic import BaseModel, ValidationError
from agent_router.core.errors import ConfigurationError


class InputValidator:
    """Validates inputs against schemas and business rules."""

    @staticmethod
    def validate_model(data: JsonObject, model_class: type[BaseModel]) -> BaseModel:
        """Validate data against a Pydantic model.

        Args:
            data: Dictionary to validate
            model_class: Pydantic model class

        Returns:
            Validated model instance

        Raises:
            ConfigurationError: If validation fails
        """
        try:
            return model_class(**data)
        except ValidationError as e:
            raise ConfigurationError(f"Validation failed: {e}")

    @staticmethod
    def validate_tool_parameters(
        parameters: JsonObject,
        required_params: List[str],
        optional_params: List[str] | None = None,
    ) -> None:
        """Validate tool parameters.

        Args:
            parameters: Parameters to validate
            required_params: List of required parameter names
            optional_params: List of optional parameter names

        Raises:
            ValueError: If required parameters are missing or unknown parameters present
        """
        # Check required parameters
        missing = set(required_params) - set(parameters.keys())
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

        # Check for unknown parameters
        if optional_params is not None:
            allowed = set(required_params) | set(optional_params)
            unknown = set(parameters.keys()) - allowed
            if unknown:
                raise ValueError(f"Unknown parameters: {unknown}")

    @staticmethod
    def sanitize_file_path(path: str) -> str:
        """Sanitize file path to prevent directory traversal attacks.

        Args:
            path: File path to sanitize

        Returns:
            Sanitized path

        Raises:
            ValueError: If path contains suspicious patterns
        """
        # Check for directory traversal attempts
        if ".." in path or path.startswith("/"):
            raise ValueError(f"Suspicious file path: {path}")

        return path.strip()
