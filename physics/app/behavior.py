"""Scripted arbitration for the world objects: what to stimulate, where to steer, when to eat.

Priority: fear (scary in range) > lust (mate in range) > hunger (food in range).
Only the winning branch's drives are on. Nothing here is a brain decision: the brain only
receives the stimulation rates, and its motor output only scales the drive (see plan()).

Units: mm, seconds (simulation time for dynamics), radians internally.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

from . import obstacles as obst
from . import odor as odor_mod
from .world import ARENA_HALF_X, ARENA_HALF_Y, RADIUS_MM, WorldStore, angular_size, clamp, falloff, heading_from_quat, sense, wrap_pi

# --- ranges and geometry (mm) -------------------------------------------------------------
FOOD_RANGE = 60.0  # food within this distance is attractive
REACH = 3.0  # thorax-to-food distance at which the fly is "touching" (feeds)
MATE_RANGE = 80.0  # a mate within this distance is pursued
MATE_CLOSE = 12.0  # inside this the receptivity dial decides: slow/stop or walk away
SCARY_RANGE = 60.0  # a scary object within this distance triggers flight
ESCAPE_DIST = 10.0  # very close: the escape giant-fibre group is driven as well

# --- stimulation rates (Hz, brain defaults from groups.json) -------------------------------
SUGAR_HZ = 200.0  # sugar_grn: food at full rate, mate at half rate
FEED_HZ = 100.0  # feed_mn9: food only, while touching
LOOM_HZ = 200.0  # looming (LC4)
LPLC2_HZ = 100.0  # lplc2
ESCAPE_HZ = 100.0  # escape_gf (giant fibre), only when very close
G_REF = 0.5  # rad/s of angular growth that drives looming at full rate

# --- thresholds ----------------------------------------------------------------------------
FEAR_MIN = 0.2  # fear dial below this: scary objects are ignored
LUST_MIN = 0.2  # lust dial below this: mates are ignored
HUNGER_MIN = 0.1  # hunger dial * hunger_level below this: food is ignored
FLEE_HOLD = 1.5  # once fleeing, keep fleeing until this multiple of SCARY_RANGE (no flip-flop at the edge)
RECEPT_AWAY = 0.25  # receptivity below this: she walks away from a male within MATE_RANGE

# --- steering and drive (scripted command) -------------------------------------------------
TURN_SCALE_DEG = 45.0  # bearing error that gives full turn
APPROACH_FWD = 1.0  # forward command when approaching food or mate (x gain 0.5..1)
FLEE_FWD = 1.0  # forward command when fleeing
EXPLORE_FWD = 0.8  # forward command with objects around but none in range
BRAIN_FLOOR = 0.5  # the brain's motor output scales the forward command between this and 1.0

# --- eating and hunger (simulation time) ---------------------------------------------------
EAT_MAX_SPEED = 2.0  # mm/s: eating needs the fly nearly stopped
EAT_UNITS_PER_S = 10.0  # food units eaten per second
HUNGER_PER_UNIT = 0.004  # hunger_level dropped per food unit eaten
HUNGER_RISE_PER_S = 0.01  # hunger_level rise per simulated second
HUNGER_START = 0.6
ARENA_MARGIN = 35.0  # mm: inside this distance of an edge the explorer is steered back toward the middle
WANDER_TURN = 0.6  # max turn command of the idle wander
SMELL_MIN = 0.01  # mean antennal concentration (units of odor.K_HALF) above which she follows the food smell
SMELL_GAIN = 150.0  # turn per unit left/right asymmetry (cL-cR)/(cL+cR); the antennae are ~1 mm apart, so asymmetry is ~0.5%/mm of gradient
SMELL_FWD = 0.9  # forward command while tracking a smell
LOOM_TAU_S = 0.3  # growth estimate smoothing: keeps looming steady while approaching, decays when still


@dataclass
class Sensed:
    id: int
    kind: str
    dist: float  # mm, fly thorax to object
    bearing: float  # rad, positive = left
    growth: float = 0.0  # rad/s, apparent angular size growth (scary only)
    amount: float | None = None


@dataclass
class Plan:
    mode: str  # idle | explore | eat | chase | flee
    forward: float = 0.0  # 0..1, before the clamp in command.py
    turn: float = 0.0  # -1..1, positive = left
    drives: dict = field(default_factory=dict)  # brain group -> rate Hz (non-zero only)
    eat_target: int | None = None  # food id the fly is touching
    flee_id: int | None = None  # scary object the fly is fleeing from (latched by Behavior)
    goal: tuple | None = None  # (bearing rad, dist mm, object id | None) the steering aims at; used by the obstacle filter


def brain_gain(brain_forward: float | None) -> float:
    """Scale for the forward command from the brain's motor output (0..1). None = no brain."""
    if brain_forward is None:
        return BRAIN_FLOOR
    return BRAIN_FLOOR + (1.0 - BRAIN_FLOOR) * clamp(float(brain_forward), 0.0, 1.0)


def _nearest(sensed, kind, max_dist, oid=None):
    best = None
    for s in sensed:
        if s.kind == kind and (oid is None or s.id == oid) and s.dist <= max_dist \
                and (best is None or s.dist < best.dist):
            best = s
    return best


DRIVE_MIN_HZ = 0.5  # rates below this are not sent (zero)


def _drives(**rates: float) -> dict:
    return {k: round(v, 2) for k, v in rates.items() if v >= DRIVE_MIN_HZ}


def plan(sensed: list[Sensed], personality: dict, hunger_level: float, brain_forward: float | None = None,
         flee_latch: int | None = None) -> Plan:
    """Arbitrate one decision from geometric sensing. Pure function; the flee latch is the only memory."""
    gain = brain_gain(brain_forward)
    if not sensed:
        return Plan("explore", forward=EXPLORE_FWD * gain)
    fear = personality["fear"]
    lust = personality["lust"]
    hunger_gain = personality["hunger"] * hunger_level

    scary = _nearest(sensed, "scary", SCARY_RANGE)
    if scary is None and flee_latch is not None:
        scary = _nearest(sensed, "scary", SCARY_RANGE * FLEE_HOLD, flee_latch)
    if scary is not None and fear >= FEAR_MIN:
        p = _flee(scary, fear, gain)
        p.flee_id = scary.id
        return p

    mate = _nearest(sensed, "mate", MATE_RANGE)
    if mate is not None and lust >= LUST_MIN:
        return _court(mate, lust, personality["receptivity"], gain)

    food = _nearest(sensed, "food", FOOD_RANGE)
    if food is not None and hunger_gain >= HUNGER_MIN and (food.amount is None or food.amount > 0):
        return _eat(food, hunger_gain, gain)

    return Plan("explore", forward=EXPLORE_FWD * gain)


def _flee(s: Sensed, fear: float, gain: float) -> Plan:
    grow = clamp(s.growth / G_REF, 0.0, 1.0)  # approaching only; receding gives nothing
    drives = _drives(looming=LOOM_HZ * fear * grow, lplc2=LPLC2_HZ * fear * grow)
    if s.dist < ESCAPE_DIST:
        drives.update(_drives(escape_gf=ESCAPE_HZ * fear))
    away = wrap_pi(s.bearing + math.pi)  # turn to face away from the object
    return Plan("flee", FLEE_FWD * gain, steer(away), drives, goal=(away, math.inf, None))


def _court(s: Sensed, lust: float, receptivity: float, gain: float) -> Plan:
    # Stand-in courtship drive: sugar_grn at half rate, feed_mn9 never driven for a mate.
    drives = _drives(sugar_grn=0.5 * SUGAR_HZ * lust * falloff(s.dist, MATE_CLOSE, MATE_RANGE))
    if receptivity < RECEPT_AWAY:  # not receptive: she walks away (a threshold on the dial, so no flip-flop)
        away = wrap_pi(s.bearing + math.pi)
        return Plan("flee", APPROACH_FWD * gain, steer(away), drives, goal=(away, math.inf, None))
    if s.dist > MATE_CLOSE:
        return Plan("chase", APPROACH_FWD * gain, steer(s.bearing), drives, goal=(s.bearing, s.dist, s.id))
    # receptive and close: slow down as she gets close, stop at receptivity 1
    return Plan("chase", APPROACH_FWD * gain * (1.0 - receptivity), steer(s.bearing), drives, goal=(s.bearing, s.dist, s.id))


def _eat(s: Sensed, hunger_gain: float, gain: float) -> Plan:
    touching = s.dist <= REACH
    drives = _drives(sugar_grn=SUGAR_HZ * hunger_gain * falloff(s.dist, REACH, FOOD_RANGE))
    if touching:
        drives.update(_drives(feed_mn9=FEED_HZ * hunger_gain))
    # arrival: slow down over the last ~12 mm so she stops on the food instead of orbiting it (the brain's walk output lags)
    forward = 0.0 if touching else APPROACH_FWD * gain * clamp((s.dist - REACH) / 12.0, 0.12, 1.0)
    return Plan("eat", forward, steer(s.bearing), drives, eat_target=s.id if touching else None, goal=(s.bearing, s.dist, s.id))


def wander(t: float) -> float:
    """Slow smooth wander: two incommensurate sines, so no random state and it never repeats exactly."""
    return WANDER_TURN * (0.65 * math.sin(2 * math.pi * t / 7.3) + 0.35 * math.sin(2 * math.pi * t / 3.1 + 1.0))


def edge_weight(x: float, y: float) -> float:
    """0 in the middle of the counter, rising to 1 at the edge (inside the soft margin)."""
    ex = (abs(x) - (ARENA_HALF_X - ARENA_MARGIN)) / ARENA_MARGIN
    ey = (abs(y) - (ARENA_HALF_Y - ARENA_MARGIN)) / ARENA_MARGIN
    return clamp(max(ex, ey), 0.0, 1.0)


def steer(bearing_rad: float) -> float:
    """Bearing rule: proportional turn toward a bearing (positive = left), full at TURN_SCALE_DEG."""
    return clamp(math.degrees(bearing_rad) / TURN_SCALE_DEG, -1.0, 1.0)


class Behavior:
    """Holds the dynamic state (hunger, speed, looming history) and runs one tick per call."""

    def __init__(self, store: WorldStore, hunger_start: float = HUNGER_START) -> None:
        self.store = store
        self.hunger_level = hunger_start
        self.speed = 0.0  # mm/s, smoothed thorax speed
        self._t: float | None = None
        self._xy: tuple[float, float] | None = None
        self._theta: dict[int, float] = {}
        self._growth: dict[int, float] = {}
        self._flee_id: int | None = None
        self.odor = odor_mod.OdorField()
        self.smell: dict = {"food": {"left": 0.0, "right": 0.0}, "cva": {"left": 0.0, "right": 0.0}}
        self.obstacle_sense: obst.ObstacleSense | None = None  # set by main.py (kitchen props); None = no obstacles
        self.avoider = obst.Avoider()
        self.rays: dict | None = None

    def reset_history(self) -> None:
        self._t = None
        self._xy = None
        self._theta.clear()
        self._growth.clear()
        self._flee_id = None
        self.speed = 0.0
        self.avoider.reset()

    def tick(self, frame: dict, thorax: int, brain_forward: float | None = None) -> dict:
        """One decision from a pose frame {t, pos, quat}. Eats and ages hunger by the sim time elapsed."""
        t = float(frame["t"])
        if self._t is not None and t < self._t:  # simulation was reset
            self.reset_history()
        first = self._t is None
        dt = 0.0 if first else t - self._t
        xy = (float(frame["pos"][thorax][0]), float(frame["pos"][thorax][1]))
        heading = heading_from_quat(frame["quat"][thorax])
        if first:
            self._t, self._xy = t, xy
        elif dt > 1e-9:
            v = math.hypot(xy[0] - self._xy[0], xy[1] - self._xy[1]) / dt
            self.speed = 0.5 * self.speed + 0.5 * v
            self._t, self._xy = t, xy
            self.hunger_level = min(1.0, self.hunger_level + HUNGER_RISE_PER_S * dt)

        objects = self.store.objects()
        sensed = []
        live = set()
        for o in objects:
            live.add(o.id)
            dist, bearing = sense(xy, heading, (o.x, o.y))
            growth = 0.0
            if o.kind == "scary":
                theta = angular_size(RADIUS_MM["scary"], dist)
                if dt > 1e-9:  # growth only from new sim time; a repeated frame keeps the last estimate
                    inst = (theta - self._theta[o.id]) / dt if o.id in self._theta else 0.0
                    alpha = 1.0 - math.exp(-dt / LOOM_TAU_S)
                    prev = self._growth.get(o.id, 0.0)
                    self._growth[o.id] = prev + alpha * (inst - prev)
                    self._theta[o.id] = theta
                elif o.id not in self._theta:
                    self._theta[o.id] = theta
                growth = self._growth.get(o.id, 0.0)
            sensed.append(Sensed(o.id, o.kind, dist, bearing, growth, o.amount))
        for oid in list(self._theta):
            if oid not in live:
                self._theta.pop(oid, None)
                self._growth.pop(oid, None)

        # smell: sources emit, the field diffuses over the elapsed simulated time, the antennae sample it
        self.odor.set_sources("food", [(o.x, o.y, 1.0 if o.amount is None else clamp(o.amount / odor_mod.FOOD_FULL, 0.05, 1.0))
                                       for o in objects if o.kind == "food" and (o.amount is None or o.amount > 0)])
        self.odor.set_sources("cva", [(o.x, o.y, 1.0) for o in objects if o.kind == "mate"])
        self.odor.step(dt)
        self.smell = {sp: self.odor.antennae(sp, xy, heading) for sp in odor_mod.SPECIES}

        p = plan(sensed, self.store.personality, self.hunger_level, brain_forward, self._flee_id)
        self._flee_id = p.flee_id
        if p.mode == "explore":  # wander, and turn back toward the middle near the edge of the counter
            w = edge_weight(xy[0], xy[1])
            _, to_centre = sense(xy, heading, (0.0, 0.0))
            p.turn = clamp((1.0 - w) * wander(t) + w * steer(to_centre), -1.0, 1.0)
        hungry = self.store.personality["hunger"] * self.hunger_level >= HUNGER_MIN
        cl, cr = self.smell["food"]["left"], self.smell["food"]["right"]
        if p.mode == "explore" and hungry and (cl + cr) / 2 >= SMELL_MIN:  # beyond sight range: follow the odour, bilaterally
            asym = (cl - cr) / (cl + cr)
            p.mode = "smell"
            p.forward = SMELL_FWD * brain_gain(brain_forward)
            p.turn = clamp(SMELL_GAIN * asym, -1.0, 1.0)
        # Receptor neurons are always on, whatever she decides. The food ORNs (ORN_DM1-4) are NOT sent to the brain by default:
        # measured, any drive on them (even 0.5 Hz) ignites a whole-brain runaway (~15k spikes per 32 ms frame that outlasts the
        # stimulus) in this LIF model at w_syn 0.275. cVA (DA1) is safe. Set ORN_FOOD_TO_BRAIN=1 to send them anyway.
        for sp, grp in (("food", "orn_food"), ("cva", "orn_cva")):
            if sp == "food" and os.environ.get("ORN_FOOD_TO_BRAIN") != "1":
                continue
            for side in ("left", "right"):
                hz = odor_mod.orn_rate_hz(self.smell[sp][side])
                if hz >= DRIVE_MIN_HZ:
                    p.drives[f"{grp}_{side}"] = round(hz, 2)
        avoid = self._avoid(p, xy, heading, objects)
        flee = None
        if p.mode == "flee" and p.flee_id is not None:
            sf = next((x for x in sensed if x.id == p.flee_id), None)
            if sf is not None:
                flee = {"bearing": sf.bearing, "dist": sf.dist, "grow": clamp(sf.growth / G_REF, 0.0, 1.0),
                        "fear": self.store.personality["fear"]}
        eaten = 0.0
        if p.eat_target is not None and dt > 1e-9 and self.speed < EAT_MAX_SPEED:
            obj = self.store.get(p.eat_target)
            if obj is not None and obj.amount is not None:
                eaten = min(EAT_UNITS_PER_S * dt, obj.amount)
                self.store.consume(p.eat_target, eaten)
                self.hunger_level = max(0.0, self.hunger_level - eaten * HUNGER_PER_UNIT)

        nearest = None
        if sensed:
            s = min(sensed, key=lambda s: s.dist)
            nearest = {"kind": s.kind, "dist": round(s.dist, 1), "bearing_deg": round(math.degrees(s.bearing), 1)}
        return {
            "smell": {sp: {s: round(v, 4) for s, v in d.items()} for sp, d in self.smell.items()},
            "mode": p.mode,
            "forward": p.forward,
            "turn": p.turn,
            "drives": p.drives,
            "eat_target": p.eat_target,
            "eaten": eaten,
            "hunger_level": round(self.hunger_level, 3),
            "nearest": nearest,
            "flee": flee,
            "speed": round(self.speed, 2),
            "avoid": avoid,
        }

    def _avoid(self, p: Plan, xy, heading: float, objects) -> dict | None:
        """Obstacle filter on the planned command (in place). Sensing is the ray fan only (app/obstacles.py)."""
        if self.obstacle_sense is None:
            return None
        goal = p.goal
        exclude = {f"obj{goal[2]}"} if goal and goal[2] is not None else set()
        self.rays = self.obstacle_sense.cast(xy, heading, obst.user_footprints(objects), exclude)
        gb, gd = (goal[0], goal[1]) if goal else (None, math.inf)
        p.forward, p.turn, info = self.avoider.filter(self.rays, p.forward, p.turn, gb, gd)
        return info

    def refilter(self, r: dict) -> dict:
        """Re-apply the obstacle filter to a command changed after tick() (the brain's own flee turn, brainloop.py)."""
        if self.rays is None or r.get("avoid") is None:
            return r
        out = dict(r)
        out["forward"], out["turn"], out["avoid"] = self.avoider.filter(self.rays, r["forward"], r["turn"])
        return out
