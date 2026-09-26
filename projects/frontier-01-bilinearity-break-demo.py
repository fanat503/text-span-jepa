"""
Демо почему билинейность ломается - код для показа на Kaggle 2xT4 и для Oral
Без torch, только numpy, максимально просто для не-матема
"""
import numpy as np
import math

print("=== ПОЧЕМУ БИЛИНЕЙНОСТЬ ЛОМАЕТСЯ - ДЕМО ===")
print()

# 1. Билинейно - фикс позиция
print("1. Фикс позиция - билинейно:")
print("   score = x_q * x_k * cos(delta_fixed)")
print("   delta_fixed=1 rad cos=0.54 const")
x_q, x_k = 1.0, 2.0
delta_fixed = 1.0
score = x_q * x_k * math.cos(delta_fixed)
print(f"   x_q={x_q}, x_k={x_k}, score={score:.3f}")
x_q2 = 2.0
score2 = x_q2 * x_k * math.cos(delta_fixed)
print(f"   x_q удвоили {x_q}->{x_q2}, score {score:.3f}->{score2:.3f} удвоился? {score2/score:.2f}x - ДА линейно")
print()

# 2. Контент-зависимая - не билинейно
print("2. Контент-зависимая - НЕ билинейно:")
print("   score = x_q * x_k * cos(x_q - x_k)  // угол зависит от x")
x_q, x_k = 1.0, 2.0
score = x_q * x_k * math.cos(x_q - x_k)
print(f"   x_q={x_q}, x_k={x_k}, delta=x_q-x_k={x_q-x_k}, cos={math.cos(x_q-x_k):.3f}, score={score:.3f}")
x_q2 = 2.0
score2 = x_q2 * x_k * math.cos(x_q2 - x_k)
print(f"   x_q удвоили {x_q}->{x_q2}, delta={x_q2-x_k}, cos={math.cos(x_q2-x_k):.3f}, score {score:.3f}->{score2:.3f} {score2/score:.2f}x - НЕ удвоился, не линейно")
print()

# 3. Почему нет разложения у exp - cos(a+b) != cos a + cos b
print("3. Почему нет разложения у exp:")
print("   Хотим cos(a+b) = U(a)+V(b) где U зависит только от a")
print("   a=90° b=0° cos90=0")
print("   a=0° b=90° cos90=0")
print("   a=90° b=90° cos180=-1, а 0+0=0 != -1")
a = math.radians(90)
b = math.radians(0)
print(f"   cos({math.degrees(a):.0f}+{math.degrees(b):.0f})={math.cos(a+b):.3f}")
a = math.radians(0); b = math.radians(90)
print(f"   cos({math.degrees(a):.0f}+{math.degrees(b):.0f})={math.cos(a+b):.3f}")
a = math.radians(90); b = math.radians(90)
print(f"   cos({math.degrees(a):.0f}+{math.degrees(b):.0f})={math.cos(a+b):.3f} != 0+0 -> нет разложения")
print()

# 4. SAE кирпичики и q_i линейно, а phi нет
print("4. SAE кирпичики:")
print("   x = f1*d1 + f2*d2, q = W_Q x = f1*W_Q d1 + f2*W_Q d2 = f1*q1 + f2*q2 линейно точно")
q1 = np.array([1.0,0.0])
q2 = np.array([0.0,1.0])
f1,f2 = 1.0,1.0
q = f1*q1 + f2*q2
phi1 = math.degrees(math.atan2(q1[1],q1[0]))
phi2 = math.degrees(math.atan2(q2[1],q2[0]))
phi = math.degrees(math.atan2(q[1],q[0]))
print(f"   q1=(1,0) угол {phi1}°, q2=(0,1) угол {phi2}°, q1+q2=(1,1) угол {phi}° != {phi1+phi2}°")
print("   Поэтому phi=angle(sum) != sum angle - угол нелинейно")
print()

# 5. Gate vs Phase зачем в RoPE/YaRN
print("5. Gate vs Phase в RoPE/YaRN:")
print("   q стрелка длина |q| gate, угол phi_q phase")
print("   score = |q||k| cos(phi_q-phi_k + pos_diff*theta)")
print("   Если |q|=0 score=0 независимо от угла - gate всегда есть в RoPE/YaRN")
print("   Кирпичик может удлинять gate_only 1.84 (-1.16 длина) или поворачивать phase_only 3.15 (+0.16 поворот)")
print("   Old margin 5.2->2.7 склеивает, мы разделяем")
print()

# 6. High-L0 vs low-L0 phi error
print("6. High-L0 vs low-L0:")
print("   phi=angle(sum f_i q_i) из 50 мелких по 0.02")
print("   Low-L0 8 берет только 8 самых больших, 42 по 0.02 теряются угол улетает 111.7° vs high-L0 50 ошибка 5°")
print("   Fidelity 63% vs 8-21% - поэтому нужен high-L0 50-100 как Gemma Scope 2 W80K L0_100")
print()

# 7. YaRN линеаризация
print("7. YaRN линеаризация exp(iD)~=1+iD:")
print("   D=delta*theta угол, exp(iD)=cosD+i sinD точка на окружности")
print("   Маленький угол 5°=0.087 рад cos=0.996~=1 sin=0.087~=D => 1+iD ошибка D^2/2")
for D in [0.1,1,1.57]:
    print(f"   D={D} cos={math.cos(D):.3f} vs 1 err {abs(math.cos(D)-1):.3f} sin={math.sin(D):.3f} vs D err {abs(math.sin(D)-D):.3f} {'PASS YaRN' if D<0.2 else 'FAIL RoPE 8192'}")
print()
print("=== ДЕМО ГОТОВО ДЛЯ KAGGLE 2xT4 И ORAL ===")
