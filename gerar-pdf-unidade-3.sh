find experimentos-unidade-3/analises \
  -type f -name "relatorio_*.md" -print0 |
while IFS= read -r -d '' arquivo; do
  diretorio=$(dirname "$arquivo")
  nome=$(basename "$arquivo" .md)

  (
    cd "$diretorio" || exit 1

    pandoc "${nome}.md" \
      --from=markdown+tex_math_single_backslash \
      --pdf-engine=typst \
      -V papersize=a4 \
      -o "${nome}.pdf"
  )

  echo "Processado: ${nome}.pdf"
done
