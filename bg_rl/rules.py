from __future__ import annotations

from bg_rl.state import Action, BackgammonState, MoveStep


def legal_checker_actions(state: BackgammonState, dice: tuple[int, int]) -> tuple[Action, ...]:
    """Return all legal checker-play actions for state.turn and dice.

    The generator enforces standard dice-use rules: use as many dice as
    possible, and when only one non-double die can be used, use the larger die
    if it is legal.
    """

    _validate_dice(dice)
    die_sequences = [(dice[0],) * 4] if dice[0] == dice[1] else [dice, (dice[1], dice[0])]
    actions: set[Action] = set()

    for die_sequence in die_sequences:
        actions.update(_walk_actions(state, tuple(die_sequence), ()))

    if not actions:
        return ((),)

    max_len = max(len(action) for action in actions)
    actions = {action for action in actions if len(action) == max_len}

    if dice[0] != dice[1] and max_len == 1:
        high_die = max(dice)
        high_die_actions = {action for action in actions if action[0].die == high_die}
        if high_die_actions:
            actions = high_die_actions

    return tuple(sorted(actions, key=_action_sort_key))


def apply_action(state: BackgammonState, action: Action) -> BackgammonState:
    next_state = state
    for step in action:
        next_state = _apply_step(next_state, step)
    return next_state.swap_turn()


def _walk_actions(state: BackgammonState, dice: tuple[int, ...], prefix: Action) -> set[Action]:
    if not dice or state.winner() is not None:
        return {prefix}

    legal_steps = _legal_steps_for_die(state, dice[0])
    if not legal_steps:
        return {prefix}

    actions: set[Action] = set()
    for step in legal_steps:
        actions.update(_walk_actions(_apply_step(state, step), dice[1:], prefix + (step,)))
    return actions


def _legal_steps_for_die(state: BackgammonState, die: int) -> tuple[MoveStep, ...]:
    player = state.turn
    if state.bar[player] > 0:
        step = _entry_step(state, die)
        return () if step is None else (step,)

    steps: list[MoveStep] = []
    for point, count in enumerate(state.points):
        if not _owned_by(count, player):
            continue
        to_point = _destination(player, point, die)
        if to_point is None:
            if _can_bear_off(state, point, die):
                steps.append(MoveStep(point, None, die, False))
            continue
        if _is_open(state, to_point, player):
            steps.append(MoveStep(point, to_point, die, _is_blot(state, to_point, player)))
    return tuple(steps)


def _entry_step(state: BackgammonState, die: int) -> MoveStep | None:
    player = state.turn
    to_point = 24 - die if player == 0 else die - 1
    if not _is_open(state, to_point, player):
        return None
    return MoveStep(None, to_point, die, _is_blot(state, to_point, player))


def _apply_step(state: BackgammonState, step: MoveStep) -> BackgammonState:
    player = state.turn
    points = list(state.points)
    bar = list(state.bar)
    off = list(state.off)
    sign = 1 if player == 0 else -1
    opponent = 1 - player

    if step.from_point is None:
        if bar[player] <= 0:
            raise ValueError("cannot move from an empty bar")
        bar[player] -= 1
    else:
        if not _owned_by(points[step.from_point], player):
            raise ValueError("cannot move a checker from an unowned point")
        points[step.from_point] -= sign

    if step.to_point is None:
        off[player] += 1
    else:
        if step.hit:
            if not _is_blot(state, step.to_point, player):
                raise ValueError("hit flag does not match board state")
            points[step.to_point] = 0
            bar[opponent] += 1
        if not _is_open_with_points(points, step.to_point, player):
            raise ValueError("destination is blocked")
        points[step.to_point] += sign

    return BackgammonState(
        points=tuple(points),
        bar=tuple(bar),  # type: ignore[arg-type]
        off=tuple(off),  # type: ignore[arg-type]
        turn=state.turn,
        cube=state.cube,
        score=state.score,
        match_length=state.match_length,
    )


def _can_bear_off(state: BackgammonState, from_point: int, die: int) -> bool:
    player = state.turn
    if state.bar[player] > 0 or not _all_in_home_board(state, player):
        return False

    if player == 0:
        pip_distance = from_point + 1
        if die == pip_distance:
            return True
        return die > pip_distance and not any(state.points[p] > 0 for p in range(from_point + 1, 6))

    pip_distance = 24 - from_point
    if die == pip_distance:
        return True
    return die > pip_distance and not any(state.points[p] < 0 for p in range(18, from_point))


def _all_in_home_board(state: BackgammonState, player: int) -> bool:
    if player == 0:
        return all(value <= 0 for value in state.points[6:])
    return all(value >= 0 for value in state.points[:18])


def _destination(player: int, point: int, die: int) -> int | None:
    destination = point - die if player == 0 else point + die
    if 0 <= destination < 24:
        return destination
    return None


def _owned_by(count: int, player: int) -> bool:
    return count > 0 if player == 0 else count < 0


def _is_open(state: BackgammonState, point: int, player: int) -> bool:
    return _is_open_with_points(list(state.points), point, player)


def _is_open_with_points(points: list[int], point: int, player: int) -> bool:
    count = points[point]
    return count >= -1 if player == 0 else count <= 1


def _is_blot(state: BackgammonState, point: int, player: int) -> bool:
    count = state.points[point]
    return count == -1 if player == 0 else count == 1


def _validate_dice(dice: tuple[int, int]) -> None:
    if len(dice) != 2:
        raise ValueError("dice must contain two values")
    if any(die < 1 or die > 6 for die in dice):
        raise ValueError("dice values must be between 1 and 6")


def _action_sort_key(action: Action) -> tuple[tuple[int, int, int, int], ...]:
    def point_value(point: int | None, bar_or_off: int) -> int:
        return bar_or_off if point is None else point

    return tuple(
        (
            point_value(step.from_point, 99),
            point_value(step.to_point, -1),
            step.die,
            int(step.hit),
        )
        for step in action
    )
