class StitchError(RuntimeError):
    """A recoverable pipeline error suitable for display to the user."""


class RegistrationError(StitchError):
    pass


class InpaintingUnavailableError(StitchError):
    pass


class CancelledError(StitchError):
    pass
