"""
Handlers for CLI commands.
Re-exports specific handlers for backward compatibility or imports.
"""

from mlip_autopipec.modules.cli_handlers.project import ProjectHandler
from mlip_autopipec.modules.cli_handlers.validation import ValidationHandler
from mlip_autopipec.modules.cli_handlers.workflow import WorkflowHandler


# Facade for backward compatibility if needed, though we will refactor app.py to use these directly.
class CLIHandler(ProjectHandler, ValidationHandler, WorkflowHandler):
    pass
