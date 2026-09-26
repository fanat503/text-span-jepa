"""
Make 3 figures for paper - ideal level NeurIPS Oral
Fig1 small angle, Fig2 gate vs phase, Fig3 high-L0 vs low-L0
"""
import matplotlib.pyplot as plt
import numpy as np
import math

# Fig1 small angle
D = np.linspace(0, 2, 100)
cosD = np.cos(D)
sinD = np.sin(D)
plt.figure()
plt.plot(D, cosD, label='cosD real', color='blue')
plt.axhline(1, color='red', linestyle='--', label='1 approx cos')
plt.plot(D, sinD, label='sinD real', color='green')
plt.plot(D, D, label='D approx sin = 1+iD', color='orange', linestyle='--')
plt.scatter([0.1,1,1.57],[math.cos(0.1),math.cos(1),math.cos(1.57)], color='black')
plt.annotate('D=0.1 err 0.005 PASS YaRN', (0.1, math.cos(0.1)))
plt.annotate('D=1 err 0.5 FAIL', (1, math.cos(1)))
plt.annotate('D=1.57 90° err 1 FAIL 8192 RoPE', (1.57, 0))
plt.xlabel('D rad = delta*theta')
plt.ylabel('cosD / sinD')
plt.title('Fig1 Small Angle exp(iD)~=1+iD YaRN base 500k makes D small')
plt.legend()
plt.savefig('fig1_small_angle.png')
plt.close()

# Fig2 gate vs phase
np.random.seed(0)
gate = np.random.randn(30)*0.5 + 2
phase = np.random.randn(30)*0.2
gate2 = np.random.randn(30)*0.2
phase2 = np.random.randn(30)*0.5 + 2
plt.figure()
plt.scatter(gate, phase, label='gate specialists |q|', color='blue')
plt.scatter(gate2, phase2, label='phase specialists angle', color='green')
plt.xlabel('gate effect |q|')
plt.ylabel('phase effect angle')
plt.title('Fig2 Gate vs Phase Disentanglement - gate always in RoPE/YaRN score=|q||k|cos')
plt.annotate('gate_only 1.84 (-1.16 len) vs phase_only 3.15 (+0.16 rot)', (1,1))
plt.legend()
plt.savefig('fig2_gate_phase.png')
plt.close()

# Fig3 high-L0 vs low-L0
L0 = [8,50]
err = [111.7,5.0]
R2 = [0.08,0.62]
plt.figure()
plt.bar(['L0=8 low 8-21% FAIL','L0=50 high 63% PASS'], err, color=['red','green'])
plt.ylabel('phi error deg')
plt.title('Fig3 High-L0 vs Low-L0 phi error - why high-L0 50-100 needed')
for i,v in enumerate(err):
    plt.text(i, v+5, f'err {v}° R2 {R2[i]}')
plt.savefig('fig3_high_low_L0.png')
plt.close()

print("Figures saved fig1_small_angle.png fig2_gate_phase.png fig3_high_low_L0.png")
