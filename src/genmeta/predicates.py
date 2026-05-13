"""Candidate predicate meanings over bounded worlds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .world import COLORS, SHAPES, SIZES, GraphWorld, ObjectWorld

Evaluator = Callable[[object, tuple[int, ...]], bool]


@dataclass(frozen=True)
class Meaning:
    """A simple observable predicate or relation."""

    id: str
    name: str
    arity: int
    evaluator: Evaluator
    group: str | None = None
    description: str = ""

    def holds(self, world: object, args: Sequence[int]) -> bool:
        if len(args) != self.arity:
            raise ValueError(f"{self.id} expects arity {self.arity}, got {len(args)}")
        return bool(self.evaluator(world, tuple(args)))


def color_is(color: str) -> Meaning:
    return Meaning(
        id=f"color:{color}",
        name=f"color_is({color})",
        arity=1,
        group="color",
        description=f"object color is {color}",
        evaluator=lambda world, args: isinstance(world, ObjectWorld)
        and world.entity(args[0]).color == color,
    )


def shape_is(shape: str) -> Meaning:
    return Meaning(
        id=f"shape:{shape}",
        name=f"shape_is({shape})",
        arity=1,
        group="shape",
        description=f"object shape is {shape}",
        evaluator=lambda world, args: isinstance(world, ObjectWorld)
        and world.entity(args[0]).shape == shape,
    )


def size_is(size: str) -> Meaning:
    return Meaning(
        id=f"size:{size}",
        name=f"size_is({size})",
        arity=1,
        group="size",
        description=f"object size is {size}",
        evaluator=lambda world, args: isinstance(world, ObjectWorld)
        and world.entity(args[0]).size == size,
    )


def same_color() -> Meaning:
    return Meaning(
        id="same_color",
        name="same_color(x,y)",
        arity=2,
        description="objects have the same color",
        evaluator=lambda world, args: isinstance(world, ObjectWorld)
        and world.entity(args[0]).color == world.entity(args[1]).color,
    )


def same_shape() -> Meaning:
    return Meaning(
        id="same_shape",
        name="same_shape(x,y)",
        arity=2,
        description="objects have the same shape",
        evaluator=lambda world, args: isinstance(world, ObjectWorld)
        and world.entity(args[0]).shape == world.entity(args[1]).shape,
    )


def edge() -> Meaning:
    return Meaning(
        id="edge",
        name="edge(x,y)",
        arity=2,
        evaluator=lambda world, args: isinstance(world, GraphWorld)
        and world.graph.has_edge(args[0], args[1]),
    )


def path_len_2() -> Meaning:
    def eval_path(world: object, args: tuple[int, ...]) -> bool:
        if not isinstance(world, GraphWorld):
            return False
        x, z = args
        return any(world.graph.has_edge(x, y) and world.graph.has_edge(y, z) for y in world.ids)

    return Meaning(id="path_len_2", name="path_len_2(x,z)", arity=2, evaluator=eval_path)


def reachable_bounded(max_depth: int = 3) -> Meaning:
    def eval_reach(world: object, args: tuple[int, ...]) -> bool:
        if not isinstance(world, GraphWorld):
            return False
        x, z = args
        if x == z:
            return True
        frontier = {x}
        seen = {x}
        for _ in range(max_depth):
            nxt: set[int] = set()
            for node in frontier:
                nxt.update(world.graph.successors(node))
            if z in nxt:
                return True
            frontier = nxt - seen
            seen.update(nxt)
        return False

    return Meaning(
        id=f"reachable:{max_depth}",
        name=f"reachable_bounded(x,z,{max_depth})",
        arity=2,
        evaluator=eval_reach,
    )


def has_self_loop() -> Meaning:
    return Meaning(
        id="self_loop",
        name="has_self_loop(x)",
        arity=1,
        evaluator=lambda world, args: isinstance(world, GraphWorld)
        and world.graph.has_edge(args[0], args[0]),
    )


def out_degree_ge(k: int) -> Meaning:
    return Meaning(
        id=f"out_degree_ge:{k}",
        name=f"out_degree_ge(x,{k})",
        arity=1,
        group="out_degree_ge",
        evaluator=lambda world, args: isinstance(world, GraphWorld)
        and world.graph.out_degree(args[0]) >= k,
    )


def object_meanings() -> list[Meaning]:
    return [*(color_is(c) for c in COLORS), *(shape_is(s) for s in SHAPES), *(size_is(z) for z in SIZES), same_color(), same_shape()]


def graph_meanings() -> list[Meaning]:
    return [edge(), path_len_2(), reachable_bounded(3), has_self_loop(), out_degree_ge(1), out_degree_ge(2)]


def meanings_for_world(world_type: str) -> list[Meaning]:
    if world_type == "object":
        return object_meanings()
    if world_type == "graph":
        return graph_meanings()
    raise ValueError(f"unknown world_type: {world_type}")
