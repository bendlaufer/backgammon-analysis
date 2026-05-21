from __future__ import annotations

from dataclasses import dataclass
import re

from bg_rl.state import Action, BackgammonState, CubeState, MoveStep


HEADER_RE = re.compile(r'^\s*;\s*\[(?P<key>[^\s]+)\s+"(?P<value>.*)"\]\s*$')
MATCH_LENGTH_RE = re.compile(r"^\s*(?P<length>\d+)\s+point match\s*$")
MOVE_LINE_RE = re.compile(r"^\s*(?P<num>\d+)\)\s*(?P<body>.*)$")
DICE_RE = re.compile(r"(?P<dice>[1-6]{2}):")
MOVE_RE = re.compile(r"(?P<src>25|2[0-4]|1[0-9]|[1-9])/(?P<dst>2[0-4]|1[0-9]|[0-9])(?P<hit>\*)?")
WIN_RE = re.compile(r"Wins\s+(?P<points>\d+)\s+points?", re.IGNORECASE)


@dataclass(frozen=True)
class CheckerDecision:
    game_index: int
    turn_number: int
    player: int
    dice: tuple[int, int]
    state: BackgammonState
    action: Action
    legal_action_count: int
    raw: str


@dataclass(frozen=True)
class ParsedMatch:
    headers: dict[str, str]
    decisions: tuple[CheckerDecision, ...]
    game_outcomes: tuple[GameOutcome, ...]
    final_score: tuple[int, int]
    match_length: int | None


@dataclass(frozen=True)
class GameOutcome:
    game_index: int
    winner: int
    points_won: int
    score_after_game: tuple[int, int]


def parse_match_text(text: str, *, validate: bool = True) -> ParsedMatch:
    """Parse an Arkadium/Jellyfish-style .mat log into checker decisions."""

    headers = _parse_headers(text)
    match_length = _parse_match_length(text)
    score = [0, 0]
    state: BackgammonState | None = None
    game_index = 0
    decisions: list[CheckerDecision] = []
    game_outcomes: list[GameOutcome] = []
    current_turn_number = 0

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("Game "):
            game_index += 1
            state = BackgammonState.initial(turn=0)
            if match_length is not None:
                state = _with_match_context(state, score, match_length)
            current_turn_number = 0
            continue

        move_line = MOVE_LINE_RE.match(raw_line)
        if move_line and state is not None:
            current_turn_number = int(move_line.group("num"))
            for player, token in _split_checker_tokens(raw_line):
                state, decision = _consume_token(
                    state,
                    token,
                    player,
                    game_index,
                    current_turn_number,
                    validate,
                )
                if decision is not None:
                    decisions.append(decision)
            continue

        win_match = WIN_RE.search(stripped)
        if win_match and state is not None:
            winner = 1 - state.turn
            points_won = int(win_match.group("points"))
            score[winner] += points_won
            game_outcomes.append(
                GameOutcome(
                    game_index=game_index,
                    winner=winner,
                    points_won=points_won,
                    score_after_game=(score[0], score[1]),
                )
            )
            state = None

    return ParsedMatch(
        headers=headers,
        decisions=tuple(decisions),
        game_outcomes=tuple(game_outcomes),
        final_score=(score[0], score[1]),
        match_length=match_length,
    )


def _consume_token(
    state: BackgammonState,
    token: str,
    player: int,
    game_index: int,
    turn_number: int,
    validate: bool,
) -> tuple[BackgammonState, CheckerDecision | None]:
    token = token.strip()
    if not token:
        return state, None
    if token.startswith("Doubles"):
        cube_value = int(token.rsplit(" ", 1)[-1])
        return _with_cube(state, CubeState(value=cube_value, owner=1 - player)), None
    if token.startswith("Takes"):
        return _with_turn(state, 1 - state.turn), None
    if token.startswith("Drops") or token.startswith("Passes"):
        return _with_turn(state, 1 - state.turn), None

    dice_match = DICE_RE.match(token)
    if dice_match is None:
        return state, None

    dice_text = dice_match.group("dice")
    dice = (int(dice_text[0]), int(dice_text[1]))
    if state.turn != player:
        state = _with_turn(state, player)

    action = _parse_action(token[dice_match.end() :], player, dice)
    legal_action_count = -1
    if validate:
        legal_actions = state.legal_actions(dice)
        legal_action_count = len(legal_actions)
        if action not in legal_actions:
            action = _matching_legal_action(action, legal_actions) or action
        if action not in legal_actions and not _listed_steps_are_legal(state, action):
            raise ValueError(
                f"illegal action in game {game_index}, turn {turn_number}, player {player}: "
                f"{token!r}; parsed={action}; legal_count={len(legal_actions)}"
            )

    decision = CheckerDecision(
        game_index=game_index,
        turn_number=turn_number,
        player=player,
        dice=dice,
        state=state,
        action=action,
        legal_action_count=legal_action_count,
        raw=token,
    )
    return state.apply_action(action), decision


def _split_checker_tokens(line: str) -> tuple[tuple[int, str], ...]:
    matches = list(DICE_RE.finditer(line))
    if not matches:
        tokens: list[tuple[int, str]] = []
        if "Doubles" in line:
            cube_index = line.index("Doubles")
            player = 0 if cube_index < 24 else 1
            response_indexes = [
                index
                for word in ("Takes", "Drops", "Passes")
                if (index := line.find(word, cube_index)) != -1
            ]
            end = min(response_indexes) if response_indexes else len(line)
            tokens.append((player, line[cube_index:end].strip()))
        if "Takes" in line:
            take_index = line.index("Takes")
            player = 0 if take_index < 24 else 1
            tokens.append((player, "Takes"))
        if "Drops" in line:
            drop_index = line.index("Drops")
            player = 0 if drop_index < 24 else 1
            tokens.append((player, "Drops"))
        if "Passes" in line:
            pass_index = line.index("Passes")
            player = 0 if pass_index < 24 else 1
            tokens.append((player, "Passes"))
        return tuple(sorted(tokens, key=lambda item: line.find(item[1])))

    right_column_start = 24
    tokens: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
        token = line[match.start() : end].strip()
        player = 0 if match.start() < right_column_start else 1
        tokens.append((player, token))

    for cube_word in ("Doubles", "Takes", "Drops", "Passes"):
        if cube_word in line:
            cube_index = line.index(cube_word)
            player = 0 if cube_index < right_column_start else 1
            token = cube_word if cube_word != "Doubles" else line[cube_index:].strip()
            tokens.append((player, token))

    return tuple(sorted(tokens, key=lambda item: line.find(item[1])))


def _parse_action(move_text: str, player: int, dice: tuple[int, int]) -> Action:
    steps: list[MoveStep] = []
    remaining_dice = [dice[0], dice[0], dice[0], dice[0]] if dice[0] == dice[1] else [dice[0], dice[1]]
    for match in MOVE_RE.finditer(move_text):
        source = int(match.group("src"))
        destination = int(match.group("dst"))
        from_point = _notation_to_internal(source, player, is_source=True)
        to_point = _notation_to_internal(destination, player, is_source=False)
        if from_point is None:
            die = 25 - destination
        elif to_point is None:
            die = source if source in remaining_dice else max(remaining_dice)
        else:
            die = abs(source - destination)
        if die in remaining_dice:
            remaining_dice.remove(die)
        steps.append(MoveStep(from_point, to_point, die, bool(match.group("hit"))))
    return tuple(steps)


def _matching_legal_action(action: Action, legal_actions: tuple[Action, ...]) -> Action | None:
    action_shape = tuple((step.from_point, step.to_point, step.hit) for step in action)
    for legal_action in legal_actions:
        legal_shape = tuple((step.from_point, step.to_point, step.hit) for step in legal_action)
        if legal_shape == action_shape:
            return legal_action
    return None


def _listed_steps_are_legal(state: BackgammonState, action: Action) -> bool:
    current = state
    for step in action:
        legal_single_steps = {
            legal_step
            for legal_action in current.legal_actions((step.die, step.die))
            for legal_step in legal_action[:1]
        }
        if step not in legal_single_steps:
            return False
        current = current.apply_action((step,))
        current = _with_turn(current, state.turn)
    return True


def _notation_to_internal(point: int, player: int, *, is_source: bool) -> int | None:
    if point == 25 and is_source:
        return None
    if point == 0 and not is_source:
        return None
    if not 1 <= point <= 24:
        raise ValueError(f"invalid point notation: {point}")
    return point - 1 if player == 0 else 24 - point


def _parse_headers(text: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in text.splitlines():
        match = HEADER_RE.match(line)
        if match:
            headers[match.group("key")] = match.group("value")
    return headers


def _parse_match_length(text: str) -> int | None:
    for line in text.splitlines():
        match = MATCH_LENGTH_RE.match(line)
        if match:
            return int(match.group("length"))
    return None


def _with_turn(state: BackgammonState, turn: int) -> BackgammonState:
    return BackgammonState(
        points=state.points,
        bar=state.bar,
        off=state.off,
        turn=turn,
        cube=state.cube,
        score=state.score,
        match_length=state.match_length,
    )


def _with_cube(state: BackgammonState, cube: CubeState) -> BackgammonState:
    return BackgammonState(
        points=state.points,
        bar=state.bar,
        off=state.off,
        turn=state.turn,
        cube=cube,
        score=state.score,
        match_length=state.match_length,
    )


def _with_match_context(
    state: BackgammonState, score: list[int], match_length: int
) -> BackgammonState:
    return BackgammonState(
        points=state.points,
        bar=state.bar,
        off=state.off,
        turn=state.turn,
        cube=state.cube,
        score=(score[0], score[1]),
        match_length=match_length,
    )
