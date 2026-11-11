"""Bounded regional search and independent, source-bound quality arbitration."""
import math
import time
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from ..contracts import Finding, Review
from ..parallel_store import BudgetExceeded, key


class MinorDeviation(Finding):
    category: Literal['font', 'layout', 'graphics']
    reason_to_accept: str = Field(min_length=1, max_length=1000)


class QualityReview(Review):
    minor_deviations: list[MinorDeviation] = Field(default_factory=list, max_length=100)

    @model_validator(mode='after')
    def minor_is_visual_only(self):
        # Content, ownership, residual and real-edit failures are supplied by
        # deterministic gates, independently of this visual assessment.
        if any(row.aspect == 'italic_only' for row in self.minor_deviations):
            raise ValueError('italic exemption is not a new minor quality waiver')
        return self


QUALITY_PROMPT = '''本次采用足够好即停止策略。仍须逐项完成全部六维检查。
findings 只列出影响实际使用的明显错误，例如内容或标点错误、旧文字残影、邻居损坏、严重裁切、溢出、结构错误。出现这些问题时 passed=false。
风格接近的字体、轻微字重字距行距、普通用户不易察觉的背景色或抗锯齿、无阅读影响的小位置差异允许接受。把这些差异写入 minor_deviations，填写具体 node_ids、description、reason_to_accept 和证据位置，passed=true 且 findings=[]。没有差异时 minor_deviations=[]。
只根据源图、最终实际 Office 图和删除编辑证据判断。不得凭自动指标声称无残影。内容、安全归属和实际编辑硬性门禁永远不能由轻微差异判断豁免。不要为了理论像素最优建议继续生成候选。'''


def quality_status(local_pass, visual):
    if not local_pass:
        return 'HARD_GATE_FAILED'
    if visual.get('status') == 'UNKNOWN':
        return 'UNKNOWN'
    if visual.get('passed') is not True or visual.get('findings'):
        return 'REVIEW_REQUIRED'
    return 'ACCEPT_WITH_MINOR_DEVIATION' if visual.get('minor_deviations') else 'ACCEPT'


DEFAULTS = {'region_candidates': 2, 'complex_region_candidates': 4,
            'repair_rounds': 1,
            'model_calls': 12, 'office_validations': 60,
            'max_seconds': 2400, 'cost_units': 120.0}
WEIGHTS = {'model_calls': 5.0, 'office_validations': 1.0, 'background_candidates': .05, 'compiler_calls': 0.0}


class AdaptiveBudget:
    """SQLite reservations survive interruption and concurrent model branches.

    Cost units are configurable operation weights, not an asserted currency
    price. Exhaustion stops new work while keeping completed regional receipts.
    """
    def __init__(self, store, options=None):
        self.store = store
        self.limits = {**DEFAULTS, **(options or {})}
        if set(self.limits) != set(DEFAULTS) or any(
                not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 0
                for v in self.limits.values()):
            raise ValueError('invalid adaptive budget')
        if any(self.limits[name]<=0 for name in ('region_candidates','complex_region_candidates','max_seconds')):
            raise ValueError('region and execution time budgets must be positive')
        for name in ('region_candidates','complex_region_candidates','repair_rounds','model_calls','office_validations'):
            if int(self.limits[name]) != self.limits[name]:
                raise ValueError('budget count must be an integer')
        with store.db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS adaptive_usage (fingerprint TEXT PRIMARY KEY, kind TEXT, units REAL, created REAL)')

    def reserve(self, kind, identity):
        fingerprint = key({'kind': kind, 'identity': identity})
        now = time.time()
        with self.store.db() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT 1 FROM adaptive_usage WHERE fingerprint=?', (fingerprint,)).fetchone():
                return False
            rows = db.execute('SELECT kind,units,created FROM adaptive_usage').fetchall()
            if rows and now-min(row[2] for row in rows) > self.limits['max_seconds']:
                raise BudgetExceeded('execution time budget exhausted')
            if kind in self.limits and sum(row[0] == kind for row in rows) >= self.limits[kind]:
                raise BudgetExceeded(kind+' budget exhausted')
            units = WEIGHTS.get(kind, 0)
            if sum(row[1] for row in rows)+units > self.limits['cost_units']:
                raise BudgetExceeded('execution cost-unit budget exhausted')
            db.execute('INSERT INTO adaptive_usage VALUES(?,?,?,?)', (fingerprint,kind,units,now))
        return True

    def report(self):
        with self.store.db() as db:
            rows = db.execute('SELECT kind,units,created FROM adaptive_usage').fetchall()
        return {'limits': self.limits, 'counts': {kind:sum(r[0]==kind for r in rows) for kind in WEIGHTS},
                'cost_units': sum(r[1] for r in rows), 'cost_unit_definition': WEIGHTS,
                'elapsed_seconds': time.time()-min((r[2] for r in rows),default=time.time())}


class BudgetedActor:
    def __init__(self, actor, budget):
        self.actor, self.budget = actor, budget

    def __getattr__(self, name):
        return getattr(self.actor, name)

    def ask_receipted(self, system, context, images, contract, receipt):
        if not Path(receipt).is_file():
            identity = {'model': self.model, 'system': system, 'context': context,
                        'receipt': str(receipt)}
            if not self.budget.reserve('model_calls', identity):
                raise BudgetExceeded('identical dispatch already reserved')
        if hasattr(self.actor, 'ask_receipted'):
            return self.actor.ask_receipted(system,context,images,contract,receipt)
        return self.actor.ask(system,context,images,contract)
