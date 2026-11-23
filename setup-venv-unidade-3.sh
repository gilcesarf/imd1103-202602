#!/usr/bin/env bash
set -euo pipefail

# Execute a partir de qualquer diretório. O ambiente do curso (.venv na raiz)
# nunca é alterado por este script.
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UNIT_DIR="${PROJECT_DIR}/experimentos-unidade-3"
VENV_DIR="${UNIT_DIR}/.venv"
REQUIREMENTS_FILE="${UNIT_DIR}/requirements-livro.txt"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

if [[ ! -d "${UNIT_DIR}" ]]; then
    echo "Erro: diretório não encontrado: ${UNIT_DIR}" >&2
    exit 1
fi

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
    echo "Erro: dependências não encontradas: ${REQUIREMENTS_FILE}" >&2
    exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Erro: ${PYTHON_BIN} não encontrado. Instale Python 3.11 ou defina PYTHON_BIN." >&2
    exit 1
fi

PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${PYTHON_VERSION}" != "3.11" ]]; then
    echo "Erro: esperado Python 3.11, encontrado ${PYTHON_VERSION} (${PYTHON_BIN})." >&2
    exit 1
fi

if [[ -e "${VENV_DIR}" ]]; then
    echo "Erro: ${VENV_DIR} já existe; não será apagado ou alterado." >&2
    exit 1
fi

echo "Criando ambiente isolado: ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
VENV_PYTHON="${VENV_DIR}/bin/python"

echo "Atualizando pip..."
"${VENV_PYTHON}" -m pip install --upgrade pip

echo "Instalando dependências do livro..."
"${VENV_PYTHON}" -m pip install -r "${REQUIREMENTS_FILE}"

echo "Verificando dependências..."
"${VENV_PYTHON}" -m pip check

echo "Ambiente pronto:"
"${VENV_PYTHON}" --version
echo "Ative com: source \"${VENV_DIR}/bin/activate\""
