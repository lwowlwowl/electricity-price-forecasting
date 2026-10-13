from scripts.decision_aware.summarize_global_taskloss_mixed_v1 import summarize


def _system(value):
    return {
        "macro_mean_daily_revenue": value,
        "macro_positive_day_rate": 1.0,
        "macro_min_daily_revenue": value,
        "macro_daily_net_revenue": {"d1": value, "d2": value},
        "macro_mean_daily_revenue_without_top_5_days": value,
        "top_5_days_total_revenue_share": 0.0,
        "macro_mean_daily_revenue_components": {
            "deviation_penalty": 0.0,
            "degradation_cost": 0.0,
        },
    }


def test_mixed_summary_reports_best_and_single_task_replacements():
    labels = {
        "best": "da_new0__rt_at_da_stable0__rt_stable0",
        "rtda": "da_stable0__rt_at_da_new0__rt_stable0",
        "rt": "da_stable0__rt_at_da_stable0__rt_new0",
        "stable": "da_stable0__rt_at_da_stable0__rt_stable0",
        "xgb": "da_xgboost_robust__rt_at_da_xgboost_robust__rt_xgboost_robust",
    }
    systems = {
        labels["best"]: _system(12.0),
        labels["rtda"]: _system(9.0),
        labels["rt"]: _system(8.0),
        labels["stable"]: _system(10.0),
        labels["xgb"]: _system(11.0),
    }
    report = summarize({
        "systems": systems,
        "ranking": [
            {"triplet": labels["best"]},
            {"triplet": labels["xgb"]},
            {"triplet": labels["stable"]},
            {"triplet": labels["rtda"]},
            {"triplet": labels["rt"]},
        ],
    })
    assert report["best_triplet_using_at_least_one_new_task"] == labels["best"]
    assert report["paired_best_vs_xgboost"]["mean_daily_revenue_difference"] == 1.0
    assert report["single_task_replacements"]["da"]["triplet"] == labels["best"]
