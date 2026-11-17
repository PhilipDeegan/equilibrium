# 2026-11-03 pair test step 1: two self-held bodies settled together vs each alone

pair.py = exp/2026/10/25/jigsaw/ring2.py with rings at x = --cx (same sense, axes z). R 8 a 3.2 q 4, depth 0.8
(depth-normalised kappa), --src local, dmax 0.9, dt 0.05, n 112, centres x = +-18 (ring edges ~7 cells apart), T 300.
Saves /bk/dst/2026/11/03/pair/{pair,solo_m,solo_p,pair_kfix}.npz (full state + d = kappa P and u^2 on the x axis at
t 100 / 200 / 300). compare.py: d_pair vs d_solo_m + d_solo_p.
Confound: normalisation gives the pair kappa 1.60e-2 vs solo 1.71e-2 (-7 %): works AGAINST a deeper middle.

t     d between |x|<=4: pair / solo sum   gap 4-10        u^2 between pair / solo sum   body share   centre (each, toward other / sideways)
100   0.75 / 0.73 (+0.02)                0.92 / 0.93     0.7x                         0.49         +0.0 / +-0.5
200   0.78 / 0.69 (+0.09)                0.97 / 0.88     20x                          0.49         +2.1 / +-1.2
300   1.20 / 0.76 (+0.43, saturated)     1.32 / 0.99     490x                         0.41         +2.9 / +-2.7
solos: centre drift <= 0.4, share 0.99, E/E0 0.942 at t 300 (pair 0.959).
Reading: settled together, the state between first matches the sum of each alone (t <= 100: superposition, no extra
pair structure); then it becomes pair-dependent: the middle deepens past the sum (to saturation), content drains
into the gap, and the bodies move toward each other (~+2.9 each) and sideways in opposite senses (the pair turns
about its midpoint): a bridge forms and they begin to merge. At this separation the pair does not settle.
pair_kfix (pair at the solo kappa, --kfix 1.713e-2): INVALID, the known fixed-coupling runaway (v10): d up to 6, whole
region saturated, both bodies dissolve by t 150. Needs a solo --kfix control or a source / clip rethink.
Next: larger separation (does a pair settle without merging? where is the bridge threshold?); opposite senses.
Philip: merging is legitimate sometimes (metallic dendrites form spontaneously -> short circuit). So the scan is a
regime map: bridge / short circuit vs settled exchange; unequal bodies / opposite senses as the 'potential difference'.
Philip: dendrites may need atoms (material transport), not a pure field automaton -> the analogy does not validate this
bridge; legitimacy of field-only merging open. Scan held pending that.
