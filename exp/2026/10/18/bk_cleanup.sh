#!/usr/bin/env bash
# 2026-10-18: /bk/dst deletions -- Philip-approved walk states (next, 10-12 item 6 / 10-08) + done-run caches (10-14, 10-16). ~73 GB.
set -e
cd /bk/dst/2026/10
rm -rf 03/coil 04/pair 04/pbox \
  05/coil/springcpp_N16_S1_q1_a10_r120_xi3.9_m0.8_R200_z0_wxi_rc_par \
  05/coil/springcpp_N30_S1_q1_a10_r120_xi3.9_m0.8_R200_z0_wxi_rc_par \
  05/halo 05/scf 05/scf_free \
  06/coil/springcpp_N8_S1_q1_a10_r120_xi3.9_m0.8_R200_z0_wxi_rc_par/state_t04000.npz \
  08/couple 08/refine 14/cartan 16/geom 16/cherenkov_rot-0.5.npz 16/cherenkov_rot0.5.npz
rmdir 04 08 14 16 2>/dev/null || true
du -sh /bk/dst/2026/10/*/
df -h /bk | tail -1
