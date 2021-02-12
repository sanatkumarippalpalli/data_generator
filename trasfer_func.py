from robustcontrol.utils import InternalDelay, tf, mimotf

NoModel = tf([0], [1])

G11 = tf([0.0039 ], [85, 1], deadtime=85)
G12 = NoModel
G13 = NoModel
G14 = tf([279.067728070848, 0.862100005149841], [7122.56780575453, 186.547394918102, 1],deadtime= 45)
G15 = tf([0.009], [300, 1], deadtime=100)
G16 = NoModel 
G21 = tf([0.0032], [160, 1], deadtime=85)
G22 = tf([-0.1],[ 85, 1], deadtime=55)
G23 = tf([-0.055], [160, 1], deadtime=100)
G24 = NoModel
G25 = NoModel
G26 = NoModel
G31 = tf([0.0033],[120, 1], deadtime=80)
G32 = NoModel
G33 = tf([-0.065],  [120, 1], deadtime=95)
G34 = NoModel
G35 = NoModel
G36 = NoModel

G = mimotf([
            [G11, G12, G13, G14, G15, G16], 
            [G21, G22, G23, G24, G25, G26],
            [G31, G32, G33, G34, G35, G36]
            ])
Gd = InternalDelay(G)
print(Gd)
