#!/bin/bash
# Rebuilds the coordination networks for russia1 then uae, in sequence (both stages use
# internal parallelism over time windows, so overlapping them would only cause
# contention). Needed to compute Newman assortativity per community -- the networks
# themselves were deleted during the earlier disk crises, while com_df.csv survived
# and must be reused unchanged.
# Log: rebuild_networks_chain.log

set -uo pipefail
cd /home/fbonaccorsi/coordinated_analysis/CoordGalaxy
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cb

DISK_FLOOR_KB=10485760   # 10G -- generous, there are ~900G free at launch

for DS in russia1 uae; do
  FREE_KB=$(df / --output=avail | tail -1 | tr -d ' ')
  if [ "$FREE_KB" -lt "$DISK_FLOOR_KB" ]; then
    echo "[chain] $(date '+%F %H:%M:%S') ABORT prima di $DS: solo $((FREE_KB/1024/1024))G liberi"
    exit 1
  fi

  echo "=================================================================="
  echo "[chain] $(date '+%F %H:%M:%S') START $DS ($((FREE_KB/1024/1024))G liberi)"
  echo "=================================================================="
  python3 scripts/rebuild_network_only.py --dataset "$DS"
  STATUS=$?
  if [ $STATUS -eq 0 ]; then
    echo "[chain] $(date '+%F %H:%M:%S') $DS COMPLETATO e verificato"
  else
    # non blocca il dataset successivo: uno puo' fallire la verifica delle soglie
    # senza che questo invalidi l'altro
    echo "[chain] $(date '+%F %H:%M:%S') $DS TERMINATO CON PROBLEMI (exit $STATUS) -- proseguo comunque"
  fi
done

echo "=================================================================="
echo "[chain] $(date '+%F %H:%M:%S') ALL DONE"
df -h / | tail -1
