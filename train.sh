#!/usr/bin/env bash
# Run directly on DGX after activating your CUDA-enabled PyTorch environment.
# Override the interpreter with DM_PYTHON=/absolute/path/to/env/bin/python.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Set your address here, or pass DM_NOTIFY_EMAIL=you@example.com when launching.
# Leave empty to disable notifications. Requires configured outgoing system mail.
DM_NOTIFY_EMAIL="${DM_NOTIFY_EMAIL:-liubixuan214@gmail.com}"
DM_MAIL_COMMAND=""
DM_STAGE="environment check"

notify_on_exit() {
    local status="$1"
    local outcome="completed"
    local host
    trap - EXIT
    host="$(hostname 2>/dev/null || printf 'unknown')"
    if [[ "$status" -ne 0 ]]; then
        outcome="failed"
    fi
    if ! printf '%s\n' \
        "DM_ex ${outcome} on ${host}." \
        "Stage: ${DM_STAGE}" \
        "Exit code: ${status}" \
        "Directory: ${PWD}" \
        "Results: ${PWD}/outputs/run1" \
        "See your terminal output or redirected log for details." \
        | "${DM_MAIL_COMMAND}" -s "[DM_ex] ${outcome} on ${host}" "${DM_NOTIFY_EMAIL}"; then
        echo "Warning: email notification failed; execution exit code is ${status}." >&2
    fi
    exit "$status"
}

if [[ -n "${DM_NOTIFY_EMAIL}" ]]; then
    if command -v mail >/dev/null 2>&1; then
        DM_MAIL_COMMAND="mail"
    elif command -v mailx >/dev/null 2>&1; then
        DM_MAIL_COMMAND="mailx"
    else
        echo "Warning: mail/mailx is unavailable; email notifications are disabled." >&2
    fi
    if [[ -n "${DM_MAIL_COMMAND}" ]]; then
        trap 'notify_on_exit "$?"' EXIT
    fi
fi

DM_PYTHON="${DM_PYTHON:-python}"
if ! command -v "${DM_PYTHON}" >/dev/null 2>&1; then
    echo "Python not found: ${DM_PYTHON}. Activate your environment or set DM_PYTHON." >&2
    exit 1
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
# Preserve CUDA_VISIBLE_DEVICES so you can select your permitted GPU externally.

"${DM_PYTHON}" -u - <<'PY'
import torch

if int(torch.__version__.split('.')[0]) < 2:
    raise SystemExit('This project requires PyTorch >= 2.0; select a compatible environment.')
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable; check your PyTorch environment and visible GPUs.')
print(f'PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}; '
      f'GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}')
PY

DM_STAGE="training"
"${DM_PYTHON}" -u train.py --device cuda --steps 5000 --batch-size 256 --seed 42 --output outputs/run1
DM_STAGE="sampling"
"${DM_PYTHON}" -u sample.py --device cuda --checkpoint outputs/run1/model.pt
DM_STAGE="training and sampling complete"
