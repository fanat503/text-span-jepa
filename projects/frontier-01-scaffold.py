# Frontier-01 — Tiny fp64 reference scaffold
# Режим: scaffold — ты пишешь core, я ревьюю
# Все в fp64, без torch, только numpy

import numpy as np

# ========= ТВОИ ФУНКЦИИ — РЕАЛИЗУЙ =========

def rotate_pairwise(q, theta):
    """
    q: [d]  d четное, например 64
    theta: [d//2] — углы для каждой пары (cos/sin)
    Возвращает R(theta) * q — ротация пар (q[2i], q[2i+1]) на theta[i]
    """
    raise NotImplementedError

def compute_base_score(q, k):
    """
    q,k: [d]
    base = <q,k> / sqrt(d)
    """
    raise NotImplementedError

def compute_hla_score(q, k, theta_q, theta_k, m, b_sal, b_dist):
    """
    q,k: [d]
    theta_q, theta_k: [d//2]
    m: scalar >0, exp(gate)
    b_sal, b_dist: scalars
    s_OLD_MODEL = m * <R(theta_q)q, R(theta_k)k> / sqrt(d) + b_sal + b_dist
    """
    raise NotImplementedError

def decompose_diff(q, k, theta_q, theta_k, m, b_sal, b_dist):
    """
    Возвращает dict с exact partition s_OLD_MODEL - s_base:
      base (для справки, не входит в diff)
      phase_only = (<Rq,Rk> - <q,k>)/sqrt(d)
      gate_only = (m-1)*<q,k>/sqrt(d)
      interaction = (m-1)*phase / sqrt(d)   где phase = <Rq,Rk>-<q,k>
      b_sal, b_dist
      sum = сумма всех термов diff (без base) должна == s_OLD_MODEL - s_base
    """
    raise NotImplementedError

def score_margin(score_target, score_foil):
    """margin = target - foil = log(p_target/p_foil)"""
    raise NotImplementedError

# ========= ТЕСТЫ — ДОЛЖНЫ СТАТЬ ЗЕЛЕНЫМИ =========

def test_base_recovery():
    np.random.seed(0)
    d=8
    q=np.random.randn(d)
    k=np.random.randn(d)
    s=compute_base_score(q,k)
    expected = np.dot(q,k)/np.sqrt(d)
    assert abs(s-expected) < 1e-12, f"base {s} vs {expected}"

def test_hla_identity():
    # gate=0 => m=1, theta=0 => R=I, biases 0 => s_OLD_MODEL == s_base
    np.random.seed(1)
    d=8
    q=np.random.randn(d)
    k=np.random.randn(d)
    theta=np.zeros(d//2)
    m=1.0
    s_base=compute_base_score(q,k)
    s_hla=compute_hla_score(q,k,theta,theta,m,0.0,0.0)
    assert abs(s_hla-s_base) < 1e-12, f"identity {s_hla} vs {s_base}"

def test_conservation():
    np.random.seed(2)
    d=8
    q=np.random.randn(d)
    k=np.random.randn(d)
    theta_q=np.random.randn(d//2)*0.5
    theta_k=np.random.randn(d//2)*0.5
    m=np.exp(np.random.randn()*0.5)  # >0
    b_sal=np.random.randn()*0.3
    b_dist=np.random.randn()*0.3
    s_base=compute_base_score(q,k)
    s_hla=compute_hla_score(q,k,theta_q,theta_k,m,b_sal,b_dist)
    parts=decompose_diff(q,k,theta_q,theta_k,m,b_sal,b_dist)
    total = parts['phase_only'] + parts['gate_only'] + parts['interaction'] + parts['b_sal'] + parts['b_dist']
    diff = s_hla - s_base
    err = abs(total - diff)
    assert err < 1e-10, f"conservation err {err}, total {total} vs diff {diff}"

def test_margin():
    # log(p_target/p_foil) = score_target - score_foil
    s_t=2.5
    s_f=1.0
    m=score_margin(s_t,s_f)
    assert abs(m-1.5) < 1e-12
    # exp(margin) = p_t/p_f
    assert abs(np.exp(m) - np.exp(s_t)/np.exp(s_f)) < 1e-12

def test_interaction_nonunique():
    # Пример из разбора: phase=3, base=1, m=0.05
    # Показываем что отдать interaction кому-то одному меняет вывод
    # Тут просто проверяем что interaction считается как (m-1)*phase
    d=2
    q=np.array([1.0,0.0])
    k=np.array([1.0,0.0])  # base=1/sqrt2
    # сделаем R так чтобы <Rq,Rk> = base+phase
    # для теста напрямую зададим phase=3 через q',k' — упростим через decompose
    # Мы тестируем формулу interaction, а не ротацию
    base = 1.0
    phase = 3.0
    m = 0.05
    gate_only = (m-1)*base
    inter = (m-1)*phase
    # если отдать interaction gate — gate кажется -3.8, если оставить отдельно — gate -0.95
    assert abs(gate_only - (-0.95)) < 1e-12
    assert abs(inter - (-2.85)) < 1e-12

if __name__ == "__main__":
    test_base_recovery()
    test_hla_identity()
    test_conservation()
    test_margin()
    test_interaction_nonunique()
    print("ALL TESTS PASSED")
