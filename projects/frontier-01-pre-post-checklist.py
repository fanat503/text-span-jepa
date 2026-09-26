# Pre-post checklist — что проверить до LessWrong
# Все комментарии на русском, код English
# Запускается на Kaggle, на dummy чекпоинте в этом окружении — только импорт

import torch
import numpy as np
from typing import Dict, List

# ========= 1. ERROR TEST: conservation <1e-10 =========
def test_conservation_error(model, q, k, theta_q, theta_k, m, b_sal, b_dist):
    """
    q,k: [d], theta_q/k: [d//2], m: scalar exp(gate), b_sal/b_dist: scalar
    Проверяем: sum(phase_only+gate_only+interaction+b_sal+b_dist) == s_OLD_MODEL - s_base
    """
    # s_base = <q,k>/sqrt(d)
    # s_OLD_MODEL = m*<Rq,Rk>/sqrt(d) + b_sal + b_dist
    # phase = <Rq,Rk> - <q,k>
    # phase_only = phase/sqrt(d)
    # gate_only = (m-1)*<q,k>/sqrt(d)
    # interaction = (m-1)*phase/sqrt(d)
    # total = phase_only+gate_only+interaction+b_sal+b_dist
    # diff = s_OLD_MODEL - s_base
    # err = |total - diff| должен <1e-10 на fp64
    raise NotImplementedError("реализуй как в scaffold.py, но на реальных тензорах из модели")


# ========= 2. GATES CORRELATION TEST =========
def collect_gates_for_correlation(model, dataloader, layer=6, n_examples=100):
    """
    Прогнать модель на 100 примерах по 512 токенов, сохранить gates.
    В model.py gates: gate_k, gate_v, gate_d, gate_sal, angle_q (theta)
    Сохраняем per token.
    """
    all_gates = {"gate_k": [], "gate_v": [], "gate_d": [], "gate_sal": [], "theta": []}
    # for batch in dataloader (n_examples):
    #   x = ln1(resid_pre) [B,T,1024]
    #   gate_k = tanh(W_gate_k(x)) [B,T,H]
    #   gate_v = tanh(W_gate_v(x))
    #   gate_d = tanh(W_gate_d(x))
    #   gate_sal = tanh(W_gate_sal(x))
    #   theta = W_phase_q * x -> [B,H,T,D//2] -> mean over D//2 для корреляции
    #   append
    # return dict of np arrays [N_tokens]
    raise NotImplementedError

def correlation_matrix(gates: Dict[str, np.ndarray]):
    """
    gates: dict name -> [N] 
    Считаем corr matrix 5x5
    Если corr(gate_k, theta) >0.8 => entangled, пишем в пост честно.
    Если <0.3 => disentangled.
    """
    names = list(gates.keys())
    data = np.stack([gates[n] for n in names], axis=0) # [5,N]
    corr = np.corrcoef(data) # [5,5]
    return names, corr


# ========= 3. HOOKS + TOPK FEATURES IN PHASE =========
def collect_hooks_n_examples(model, dataloader, layer=6, n=100):
    """
    Как собирать topk фич в phase если phase выдается по состоянию:
    - phase per token = <Rq,Rk> - <q,k>  [B,T,H]
    - Но нам нужен x_i = ln1(resid_pre) [B,T,1024] для SAE
    - Сохраняем x_i half + phase + margin + gates в .pt файлы батчами по 4
    - Не сохраняем логиты [B,T,50257] — слишком большие
    """
    # for i,batch in enumerate(dataloader):
    #   tokens [B,512]
    #   hook: model.transformer.h[layer].ln_1.register_forward_hook(save x_i)
    #   logits, etc = model(tokens)
    #   phase = ... из model.transformer.h[layer].attn.last_angle_q_abs_mean? 
    #   Лучше: raw_angles_q = einsum("btd,hdk->bhtk", x_float, W_phase_q) -> theta
    #   margin = log(p_target/p_foil) как раньше через topk
    #   torch.save({"x": x_i.half().cpu(), "phase": phase.cpu(), "margin": margin.cpu(), "gates": gates}, f"batch_{i}.pt")
    raise NotImplementedError

def topk_features_in_phase(W_phase, W_dec, f_active, topk=10):
    """
    W_phase: [D_head//2, D_model] или [D_model, D_head//2] — проверь shape в model.py: [n_head, n_embd, head_dim//2]
    W_dec: [n_dict, D_model] — d_k направления SAE
    f_active: [n_dict] sparse, 20-30 ненулевых на токен
    
    Как получить topk:
    1. Для каждого активного k: contrib_k = f_k * ||W_phase * d_k||  или точнее f_k * (W_phase * d_k) -> [D_head//2] -> norm
    2. Ранжируешь contrib_k по убыванию
    3. topk = argsort(contrib_k)[-10:]
    
    Если phase выдается как скаляр per token (mean angle), то:
    theta_token = mean(W_phase * x_i) ~ sum f_k*(W_phase*d_k) mean
    """
    # scores = |W_phase @ d_k|  # [n_dict] любимчики матрицы
    # contrib = f_active * scores  # [n_dict] вклад на этом токене
    # topk_idx = np.argsort(contrib)[-topk:]
    raise NotImplementedError


# ========= SAE TRAIN (стриминг) =========
class SAE(torch.nn.Module):
    def __init__(self, d_model=1024, n_dict=10000, topk=30):
        super().__init__()
        self.W_enc = torch.nn.Parameter(torch.randn(d_model, n_dict)*0.01)
        self.W_dec = torch.nn.Parameter(torch.randn(n_dict, d_model)*0.01)
        self.topk = topk
        # Нормируем W_dec строки на 1
    def encode(self, x): # x [B, D]
        f = torch.relu(x @ self.W_enc) # [B, n_dict]
        # topk sparsity
        topk_val, topk_idx = torch.topk(f, self.topk, dim=-1)
        f_sparse = torch.zeros_like(f).scatter_(-1, topk_idx, topk_val)
        return f_sparse
    def decode(self, f): # f [B, n_dict]
        return f @ self.W_dec # [B, D] = sum f_k*d_k

def train_sae_streaming(sae, dataloader_files, steps=100000):
    """
    dataloader_files: список batch_*.pt с x [B,T,D]
    Обучение online, batch 10k токенов, чтобы не 20GB RAM
    Loss = MSE(x, x_hat) + L1(f)*1e-3
    """
    raise NotImplementedError


# ========= STEERING CODE =========
def steer_with_feature(x, d_k, alpha=1.0, remove=False):
    """
    x: [D] - x_i после ln1
    d_k: [D] - направление фичи из W_dec
    alpha: сколько добавить
    remove: если True -> x - f*d_k, если False -> x + alpha*d_k
    
    Для проверки:
    - x_new = x - f_k*d_k -> theta должен упасть
    - x_new = x + 1.0*d_k в не-коде -> theta должен вырасти (counterfactual)
    """
    if remove:
        # f_k нужно знать, обычно f_k = активация этой фичи на этом токене
        return x - alpha * d_k
    else:
        return x + alpha * d_k


# ========= BIG FUNCTION: объединяет все =========
def run_full_pipeline_before_post(
    model,
    dataloader,
    sae,
    layer=6,
    n_examples=100
):
    """
    Одна большая функция которая при готовых SAE и модели делает все проверки до поста.
    Возвращает dict с результатами для вставки в LessWrong.
    """
    results = {}

    # 1. Error test
    # for one token: q,k,theta_q,theta_k,m,b_sal,b_dist -> test_conservation_error
    # results["conservation_error"] = err

    # 2. Gates correlation
    # gates = collect_gates_for_correlation(model, dataloader, layer, n_examples)
    # names, corr = correlation_matrix(gates)
    # results["corr_matrix"] = corr
    # results["corr_names"] = names

    # 3. Hooks + save
    # collect_hooks_n_examples(model, dataloader, layer, n_examples) -> файлы batch_*.pt

    # 4. Topk features in phase
    # W_phase = model.transformer.h[layer].attn.W_phase_q # [H, D, D//2]
    # W_dec = sae.W_dec # [n_dict, D]
    # Для каждого файла batch_*.pt:
    #   x [B*T, D], f = sae.encode(x) [B*T, n_dict]
    #   scores = |W_phase[head] @ W_dec.T| [n_dict]
    #   contrib = f * scores
    #   topk = topk_features_in_phase(...)
    # results["top10_favorites"] = top10 по scores (любимчики матрицы)
    # results["top10_per_token"] = top10 по contrib (на конкретных токенах)

    # 5. Steering test
    # x_code = токен где f_def активна, x_text = обычный текст
    # theta_before, theta_after_remove = ...
    # theta_before_text, theta_after_add = ...
    # results["steering_remove"] = delta theta
    # results["steering_add"] = delta theta counterfactual

    # 6. Conditional compute benefit
    # results["conditional"] = {"loss_full": X, "loss_cond": X+0.0001, "time_full": Y, "time_cond": 0.1*Y}

    return results

# Чеклист что должно быть < порога до поста:
# - conservation_error <1e-10
# - corr(gate_k, theta) <0.8 иначе пишем entangled
# - top10 favorites |W*d| : max >0.5, random ~0.02
# - steering_remove: theta 0.8->0.1, random 0.8->0.79
# - steering_add: theta 0.1->0.7 в не-коде
# - conditional: loss +0.0001, time 0.1*Y
