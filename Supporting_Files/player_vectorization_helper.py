from sklearn.cluster import KMeans
import os
import json
import numpy as np
import bisect
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc

# -----------------File Handling----------------------

matches_dir = "Data/Wyscout/matches/"

# ----------------Helper Methods---------------------

def init_player_stats():
    return {
        # Passing
        "total_passes": 0,
        "accurate_passes": 0,
        "key_passes": 0,
        "assists": 0,
        "pass_dist_total": 0,
        "forward_passes": 0,
        "backward_passes": 0,
        "lateral_passes": 0,
        "through_ball_attempts": 0,

        # Shooting
        "total_shots": 0,
        "goals": 0,
        "own_goals": 0,
        "left_shots": 0,
        "right_shots": 0,
        "head_body_shots": 0,
        "shots_on_target": 0,

        # Defensive
        "total_duels": 0,
        "duels_won": 0,
        "defensive_duels": 0,
        "interceptions": 0,
        "clearances": 0,

        # Possession
        "touches": 0,
        "takeons": 0,

        # Carrying
        "progressive_carries": 0,
        "prog_carry_dist_total": 0,

        # Positional tendencies
        "zone_entries_att_third": 0,

        # Derived metrics
        "pass_accuracy": 0.0,
        "avg_pass_length": 0.0,
        "shot_accuracy": 0.0,
        "duel_win_pct": 0.0,
        "pass_to_touch_ratio": 0.0,
        "avg_prog_carry_dist": 0.0,

        "total_goals": 0,
        "norm_avg_x": 0.0,
        "norm_avg_y": 0.0,
        "cluster" : 0,
        "team_events_on_field": 0,
        "other_team_events_on_field": 0,
        "team_total_shots": 0,
        "team_total_passes": 0,
        "team_total_takeons": 0,
        "team_progressive_carries": 0,
        "team_att_third_entries": 0,
        "team_avg_prog_carry_dist": 0.0,
        "opp_total_goals": 0,

        "position_sum_x": 0.0,
        "position_sum_y": 0.0,
        "position_count": 0
    }

def make_stat_map():
    return defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                init_player_stats
            )
        )
    )

def process_event(ev, stat_map):
    m = ev["matchId"]
    t = ev["teamId"]
    p = ev["playerId"]
    stats = stat_map[m][t][p]

    pos_list = ev.get("positions", [])
    if isinstance(pos_list, list) and len(pos_list) > 0:
        p0 = pos_list[0]
        x, y = p0.get("x"), p0.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            stats["position_sum_x"] += x
            stats["position_sum_y"] += y
            stats["position_count"] += 1

    if ev["eventName"] == "Pass":
        stats["total_passes"] += 1

        if any(tag["id"] == 302 for tag in ev.get("tags", [])):
            stats["key_passes"] += 1

        if any(tag["id"] == 301 for tag in ev.get("tags", [])):
            stats["assists"] += 1

        if any(tag["id"] == 1801 for tag in ev.get("tags", [])):
            stats["accurate_passes"] += 1

        # Pass distance
        pos_list = ev.get("positions", [])
        if isinstance(pos_list, list) and len(pos_list) == 2:
            p0 = pos_list[0]
            p1 = pos_list[1]
            x1, y1 = p0["x"], p0["y"]
            x2, y2 = p1["x"], p1["y"]
            d = ((x2-x1)**2 + (y2-y1)**2)**0.5
            stats["pass_dist_total"] += d

        if any(tag["id"] == 901 for tag in ev.get("tags", [])):
            stats["through_ball_attempts"] += 1

    if ev["eventName"] == "Shot":
        stats["total_shots"] += 1
        if any(tag["id"] == 101 for tag in ev.get("tags", [])):
            stats["goals"] += 1
        if any(tag["id"] == 102 for tag in ev.get("tags", [])):
            stats["own_goals"] += 1
        for foot_id, name in ((401,"left_shots"), (402,"right_shots"), (403,"head_body_shots")):
            if any(tag["id"] == foot_id for tag in ev.get("tags", [])):
                stats[name] += 1
        
        if any(tag["id"] in {1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209} for tag in ev.get("tags", [])):
            stats["shots_on_target"] += 1

    if ev["eventName"] == "Duel":
        stats["total_duels"] += 1
        # Won?
        if any(tag["id"] == 703 for tag in ev.get("tags", [])):
            stats["duels_won"] += 1

    if ev["subEventName"] == "Ground defending duel":
        stats["defensive_duels"] += 1

    if any(tag["id"] == 1401 for tag in ev.get("tags", [])):
        stats["interceptions"] += 1

    if any(tag["id"] == 1501 for tag in ev.get("tags", [])):
        stats["clearances"] += 1

    if ev["eventName"] == "Others on the ball" and ev["subEventName"] == "Touch":
        stats["touches"] += 1

    if any(tag["id"] in (503,504) for tag in ev.get("tags", [])):
        stats["takeons"] += 1

    if ev["eventName"] != "Pass" and len(ev.get("positions",[])) == 2:
        pos_list = ev.get("positions", [])
        if isinstance(pos_list, list) and len(pos_list) == 2:
            p0 = pos_list[0]
            p1 = pos_list[1]
            x1, y1 = p0["x"], p0["y"]
            x2, y2 = p1["x"], p1["y"]
        d = ((x2-x1)**2 + (y2-y1)**2)**0.5
        if d >= 10:
            stats["progressive_carries"] += 1
            stats["prog_carry_dist_total"] += d

    pos = ev.get("positions") or []
    if len(pos) == 2:
        end_y, end_x = pos[1]["y"], pos[1]["x"]
        if end_x >= 66:
            stats["zone_entries_att_third"] += 1

    if ev["eventName"] == "Pass" and len(pos) == 2:
        p0, p1 = pos_list         
        x1, y1 = p0["x"], p0["y"]
        x2, y2 = p1["x"], p1["y"]

        dx = x2 - x1
        if dx > 0:
            stats["forward_passes"] += 1
        elif dx < 0:
            stats["backward_passes"] += 1
        else:
            stats["lateral_passes"] += 1

def calculate_additional_stats(stat_map, goals_map, opp_goals_map):

    shots_map = defaultdict(dict)
    passes_map = defaultdict(dict)
    takeons_map = defaultdict(dict)
    progressive_carries_map = defaultdict(dict)
    att_third_entries_map = defaultdict(dict)
    avg_prog_carry_dist_map = defaultdict(dict)
    for m, teams in stat_map.items():
        for t, players in teams.items():
            for p, s in players.items():
                if s["total_passes"] > 0:
                    s["pass_accuracy"] = s.get("accurate_passes") / s["total_passes"]
                    s["avg_pass_length"] = s["pass_dist_total"] / s["total_passes"]
                else:
                    s["pass_accuracy"] = 0.0
                    s["avg_pass_length"] = 0.0

                if s["total_shots"] > 0:
                    s["shot_accuracy"] = s.get("shots_on_target") / s["total_shots"]
                else:
                    s["shot_accuracy"] = 0.0

                if s["total_duels"] > 0:
                    s["duel_win_pct"] = s["duels_won"] / s["total_duels"]
                else:
                    s["duel_win_pct"] = 0.0

                if s["touches"] > 0:
                    s["pass_to_touch_ratio"] = s["total_passes"] / s["touches"]
                else:
                    s["pass_to_touch_ratio"] = 0.0

                if s["progressive_carries"] > 0:
                    s["avg_prog_carry_dist"] = s["prog_carry_dist_total"] / s["progressive_carries"]
                else:
                    s["avg_prog_carry_dist"] = 0.0
                
                if s["position_count"] > 0:
                    s["avg_position_x"] = s["position_sum_x"] / s["position_count"]
                    s["avg_position_y"] = s["position_sum_y"] / s["position_count"]
                else:
                    s["avg_position_x"] = 0.0
                    s["avg_position_y"] = 0.0

                s["total_goals"] = goals_map[m][t]
                s["opp_total_goals"] = opp_goals_map[m][t]
                shots_map.setdefault(m, {}).setdefault(t, 0)
                shots_map[m][t] += s["total_shots"]
                takeons_map.setdefault(m, {}).setdefault(t, 0)
                takeons_map[m][t] += s["takeons"]
                passes_map.setdefault(m, {}).setdefault(t, 0)
                passes_map[m][t] += s["total_passes"]
                att_third_entries_map.setdefault(m, {}).setdefault(t, 0)
                att_third_entries_map[m][t] += s["zone_entries_att_third"]
                progressive_carries_map.setdefault(m, {}).setdefault(t, 0)
                progressive_carries_map[m][t] += s["progressive_carries"]
                avg_prog_carry_dist_map.setdefault(m, {}).setdefault(t, 0)
                avg_prog_carry_dist_map[m][t] += s["avg_prog_carry_dist"]
    
    for m, teams in stat_map.items():
        for t, players in teams.items():
            for p, s in players.items():
                s["team_total_shots"] = shots_map[m][t]
                s["team_total_passes"] = passes_map[m][t]
                s["team_total_takeons"] = takeons_map[m][t]
                s["team_progressive_carries"] = progressive_carries_map[m][t]
                s["team_att_third_entries"] = att_third_entries_map[m][t]
                s["team_avg_prog_carry_dist"] = avg_prog_carry_dist_map[m][t]
    return stat_map

def add_normalized_positions(stat_map):
    for match_id, teams in stat_map.items():
        for team_id, players in teams.items():
            coords = []
            for pid, stats in players.items():
                x = stats.get("avg_position_x")
                y = stats.get("avg_position_y")
                if x is None or y is None:
                    continue
                coords.append((pid, x, y))
            if not coords:
                continue

            gk_pid, _, _ = min(coords, key=lambda triple: triple[1])

            outfield = [(pid, x, y) for pid, x, y in coords if pid != gk_pid]
            if not outfield:
                players[gk_pid]["norm_avg_x"] = None
                players[gk_pid]["norm_avg_y"] = None
                del players[gk_pid]
                continue

            xs = [x for _, x, _ in outfield]
            ys = [y for _, _, y in outfield]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            x_den = x_max - x_min or 1.0
            y_den = y_max - y_min or 1.0

            for pid, x, y in outfield:
                players[pid]["norm_avg_x"] = (x - x_min) / x_den
                players[pid]["norm_avg_y"] = (y - y_min) / y_den

    return stat_map

def assign_position_clusters(stat_map, n_clusters=8):
    entries = []   
    coords  = []
    for match_id, teams in stat_map.items():
        for team_id, players in teams.items():
            xs, ys = [], []
            for pid, stats in players.items():
                x, y = stats.get("norm_avg_x"), stats.get("norm_avg_y")
                if x is None or y is None:
                    continue
                entries.append((match_id, team_id, pid))
                coords.append((x, y))
    coords = np.array(coords)
    
    km = KMeans(n_clusters=n_clusters, random_state=0)
    labels = km.fit_predict(coords)  

    for (match_id, team_id, pid), lbl in zip(entries, labels):
        stat_map[match_id][team_id][pid]["cluster"] = int(lbl) + 1

    return stat_map

def get_subs_map():

    subs_map      = defaultdict(lambda: defaultdict(dict))
    goals_map     = defaultdict(dict)
    opp_goals_map = defaultdict(dict)

    for fn in os.listdir(matches_dir):
        if not fn.endswith(".json"):
            continue

        path = os.path.join(matches_dir, fn)
        with open(path) as f:
            data = json.load(f)

        matches = data if isinstance(data, list) else [data]

        for match in matches:
            m = match["wyId"]
            teams = match.get("teamsData", {})
            team_ids = list(teams.keys())

            for teamId_str, tdata in teams.items():
                teamId = int(teamId_str)
                score = tdata.get("score", 0)
                goals_map[m][teamId] = score

                other = int(team_ids[1]) if teamId_str == team_ids[0] else int(team_ids[0])
                opp_goals_map[m][teamId] = teams[team_ids[0] if str(other)==team_ids[1] else team_ids[1]].get("score", 0)

                formation = tdata.get("formation", {})
                lineup = formation.get("lineup", [])
                subs   = formation.get("substitutions", [])

                for p in lineup:
                    pid = p["playerId"]
                    subs_map[m][teamId][pid] = [0, None]

                if isinstance(subs, list):
                    for sub in subs:
                        pin, pout, minute = sub["playerIn"], sub["playerOut"], sub["minute"]
                        t0 = minute * 60
                        subs_map[m][teamId][pin] = [t0, None]
                        if pout in subs_map[m][teamId]:
                            subs_map[m][teamId][pout][1] = t0
                        else:
                            subs_map[m][teamId][pout] = [0, t0]

    return subs_map, goals_map, opp_goals_map

def get_team_events(stat_map, subs_map, team_event_times,):

    match_end_time = {}
    for m, teams in team_event_times.items():
        all_ts = [ts for t in teams.values() for ts in t]
        match_end_time[m] = max(all_ts)

    for m, teams in subs_map.items():
        end_t = match_end_time.get(m)
        if end_t is None:
            continue
        for teamId, players in teams.items():
            for pid, (t_in, t_out) in players.items():
                if t_out is None:
                    players[pid][1] = end_t
    
    for m in team_event_times:
        for t in team_event_times[m]:
            team_event_times[m][t].sort()

    for m, teams in stat_map.items():
        match_team_ids = list(teams.keys())
        if len(match_team_ids) != 2:
            continue
        t1, t2 = match_team_ids

        for t, players in teams.items():
            other_t = t2 if t == t1 else t1

            times_own  = team_event_times[m].get(t,   [])
            times_opp  = team_event_times[m].get(other_t, [])

            for p, stats in players.items():
                tin, tout = subs_map[m][t].get(p, (0, match_end_time[m]))

                lo = bisect.bisect_left(times_own, tin)
                hi = bisect.bisect_right(times_own, tout)
                stats["team_events_on_field"] = hi - lo

                lo2 = bisect.bisect_left(times_opp, tin)
                hi2 = bisect.bisect_right(times_opp, tout)
                stats["other_team_events_on_field"] = hi2 - lo2
    
    return stat_map

def make_ratio(numer, denom):
    try:
        return numer / denom if denom else 0.0
    except Exception:
        return 0.0

def build_new_vector(stats):
    if stats.get("position_count", 0) < 5:
        return None

    tp      = stats.get("total_passes", 0)
    ttp     = stats.get("team_total_passes", 0)
    fp      = stats.get("forward_passes", 0)
    bp      = stats.get("backward_passes", 0)
    kp      = stats.get("key_passes", 0)
    ast     = stats.get("assists", 0)
    tg      = stats.get("total_goals", 0)
    tba     = stats.get("through_ball_attempts", 0)
    ts      = stats.get("total_shots", 0)
    tshot   = stats.get("team_total_shots", 0)
    gls     = stats.get("goals", 0)
    og      = stats.get("own_goals", 0)
    og_opp   = stats.get("opp_total_goals", 0)
    sot     = stats.get("shots_on_target", 0)
    td      = stats.get("total_duels", 0)
    ote     = stats.get("other_team_events_on_field", 0)
    dw      = stats.get("duels_won", 0)
    dd      = stats.get("defensive_duels", 0)
    inter   = stats.get("interceptions", 0)
    clr     = stats.get("clearances", 0)
    tk      = stats.get("takeons", 0)
    tkt     = stats.get("team_total_takeons", 0)
    pc      = stats.get("progressive_carries", 0)
    tpc     = stats.get("team_progressive_carries", 0)
    apd     = stats.get("avg_prog_carry_dist", 0)
    tapd    = stats.get("team_avg_prog_carry_dist", 0)
    zei     = stats.get("zone_entries_att_third", 0)
    tzei    = stats.get("team_att_third_entries", 0)
    pa      = stats.get("pass_accuracy", 0)
    sa      = stats.get("shot_accuracy", 0)
    dwp     = stats.get("duel_win_pct", 0)
    apl     = stats.get("avg_pass_length", 0)

    new_feats = {
        "t_passes":                          make_ratio(tp,  ttp),
        "pass_accuracy":                     pa,
        "f_pass_ratio":                      make_ratio(fp,  tp),
        "b_pass_ratio":                      make_ratio(bp,  tp),
        "k_pass_ratio":                      make_ratio(kp,  tp),
        "assist_ratio":                      make_ratio(ast, tg),
        "avg_pass_length":                   apl,
        "through_ball_attempt_ratio":        make_ratio(tba, tp),
        "total_shots_ratio":                 make_ratio(ts,  tshot),
        "goals_ratio":                       make_ratio(gls, tg),
        "own_goal_ratio":                    make_ratio(og,  og_opp),
        "shots_on_target_ratio":             make_ratio(sot, ts),
        "total_duel_ratio":                  make_ratio(td,  ote),
        "def_duel_ratio":                    make_ratio(dd,  ote),
        "interception_ratio":                make_ratio(inter, ote),
        "clearance_ratio":                   make_ratio(clr, ote),
        "takeon_ratio":                      make_ratio(tk,  tkt),
        "progressive_carry_ratio":           make_ratio(pc,  tpc),
        "avg_prog_carry_dist_ratio":         make_ratio(apd, tapd),
        "zone_entries_att_third_ratio":      make_ratio(zei, tzei),
        "shot_accuracy":                     sa,
        "duel_win_ratio":                    dwp,
    }

    return new_feats

def transform_stat_map(stat_map):

    new_map = {}

    for match_id, teams in stat_map.items():
        new_map[match_id] = {}
        for team_id, players in teams.items():
            new_map[match_id][team_id] = {}
            for player_id, stats in players.items():
                new_feats = build_new_vector(stats)
                if new_feats is not None:
                    new_map[match_id][team_id][player_id] = new_feats

    return new_map

def normalize_new_map(new_map):

    feat_min = {}
    feat_max = {}
    for teams in new_map.values():
        for players in teams.values():
            for stats in players.values():
                for feat, val in stats.items():
                    if feat not in feat_min or val < feat_min[feat]:
                        feat_min[feat] = val
                    if feat not in feat_max or val > feat_max[feat]:
                        feat_max[feat] = val

    norm_map = {}
    for match_id, teams in new_map.items():
        norm_map[match_id] = {}
        for team_id, players in teams.items():
            norm_map[match_id][team_id] = {}
            for player_id, stats in players.items():
                norm_stats = {}
                for feat, val in stats.items():
                    mn = feat_min[feat]
                    mx = feat_max[feat]
                    if mx > mn:
                        norm_val = (val - mn) / (mx - mn)
                    else:
                        norm_val = 0.0
                    norm_stats[feat] = norm_val
                norm_map[match_id][team_id][player_id] = norm_stats

    return norm_map

def split_feature_map_by_cluster(stat_map, feature_map, num_clusters=8):

    cluster_maps = {c: {} for c in range(1, num_clusters+1)}

    for match_id, teams in feature_map.items():
        for team_id, players in teams.items():
            for player_id, feats in players.items():
                try:
                    cluster = stat_map[match_id][team_id][player_id]["cluster"]
                except KeyError:
                    continue
                
                cmatch = cluster_maps[cluster].setdefault(match_id, {})
                cteam  = cmatch.setdefault(team_id, {})

                cteam[player_id] = feats

    return cluster_maps

def draw_pitch(ax, scale=100):
    ax.add_patch(Rectangle((0, 0), scale, scale, fill=False))
    ax.plot([scale/2, scale/2], [0, scale], color='black')
    centre_circle = Circle((scale/2, scale/2), scale*0.1, fill=False)
    ax.add_patch(centre_circle)
    for x in (0, scale):
        box_width = scale * 0.06 if x == 0 else -scale * 0.06
        ax.add_patch(Rectangle((x, scale*0.3), box_width, scale*0.4, fill=False))
        area_width = scale * 0.16 if x == 0 else -scale * 0.16
        ax.add_patch(Rectangle((x, scale*0.18), area_width, scale*0.64, fill=False))
        spot_x = scale * 0.12 if x == 0 else scale * 0.88
        arc = Arc((spot_x, scale/2), scale*0.2, scale*0.2, angle=0,
                  theta1=310 if x == 0 else 130, theta2=50 if x == 0 else 230)
        ax.add_patch(arc)
    ax.set_xlim(0, scale)
    ax.set_ylim(0, scale)
    ax.set_aspect('equal')
    ax.axis('off')

def plot_all_normalized_clusters(stat_map, scale=100):
    xs, ys, clusters = [], [], []
    for match_id, teams in stat_map.items():
        for team_id, players in teams.items():
            for pid, stats in players.items():
                c = stats.get("cluster")
                x = stats.get("norm_avg_x")
                y = stats.get("norm_avg_y")
                if c is None or x is None or y is None:
                    continue
                xs.append(x * scale)
                ys.append(y * scale)
                clusters.append(c)

    fig, ax = plt.subplots(figsize=(8, 6))
    draw_pitch(ax, scale=scale*100 if scale==1 else scale)
    scatter = ax.scatter(xs, ys, c=clusters, cmap='tab10', alpha=0.7)
    handles, _ = scatter.legend_elements(num=8)
    legend_labels = [f"Cluster {i}" for i in sorted(set(clusters))]
    ax.legend(handles, legend_labels, loc='upper right', title="Clusters")
    ax.set_title("All Matches & Teams: Normalized Positions by Cluster")
    plt.show()