from dataclasses import dataclass, field


@dataclass
class SessionState:
    user_profile: dict
    messages: list[str] = field(default_factory=list)

    def add_message(self, message: str) -> None:
        self.messages.append(message)