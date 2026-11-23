# Plano de experimentação — Aprendizado por Reforço Profundo

**Disciplina:** Aprendizado por Reforço — IMD/UFRN  
**Referências centrais:** aulas 25 e 26; capítulos 11 e 12 de *Deep Reinforcement Learning Hands-On*, 3ª edição (Maxim Lapan)  
**Data de consolidação:** 22/11/2026  
**Estado:** plano acordado; preparação do ambiente pendente; **nenhum experimento deste plano foi executado ou validado ainda**  
**Finalidade:** referência de execução, documentação de decisões e transferência de contexto para outros agentes de IA.

> **Orientação para agentes:** este documento descreve o plano e o estado atual, não resultados experimentais. Distinguir decisões já tomadas de propostas sujeitas à validação. Não alterar os algoritmos ou executar treinamentos extensos sem alinhar o objetivo da tarefa específica. Priorizar alterações pequenas e verificáveis, preservando os experimentos anteriores da disciplina.

## 1. Objetivo e escopo

Executar e analisar os exercícios indicados no final das duas aulas:

- **Aula 25 — capítulo 11:** experimentar **REINFORCE no CartPole**, com o código do livro.
- **Aula 26 — capítulo 12:** experimentar **Actor-Critic no Pong**, com o código do livro.

Organizar os exercícios como uma sequência de investigação, com extensões justificadas pela teoria e pelo código, em vez de executar exemplos sem comparação.

**Questão orientadora:** como técnicas de redução de variância dos gradientes e a utilização de uma função de valor (crítico) influenciam a convergência, a estabilidade e a eficiência do aprendizado por reforço profundo?

**Eixos de investigação:**

1. **CartPole:** REINFORCE versus DQN; efeito do baseline; medição dos gradientes.
2. **Pong:** desempenho de métodos de gradiente de política versus A2C; impacto de execução síncrona e assíncrona; paralelização dos gradientes.

O trabalho é uma atividade **experimental e didática**, não um benchmark exaustivo ou uma busca pelo estado da arte. Resultados negativos ou ausência de convergência também devem ser relatados.

## 2. Fontes e artefatos de referência

Arquivos originalmente fornecidos:

| Arquivo | Finalidade |
|---|---|
| `Aprendizado_por_Reforco_-_Aula_25_-_2026.pdf` | Fundamentação e exercício do REINFORCE no CartPole (última página). |
| `Aprendizado_por_Reforco_-_Aula_26_-_2026.pdf` | Fundamentação de Actor-Critic/A2C/A3C e exercício do Pong (última página). |
| `Deep_Reinforcement_Learning_Hands_on_-_Third_Edition_-_Capitulo_11.pdf` | REINFORCE, baseline, entropia, PG no CartPole/Pong, comparação com DQN. |
| `Deep_Reinforcement_Learning_Hands_on_-_Third_Edition_-_Capitulo_12.pdf` | Variância, A2C no Pong e paralelização/A3C. |
| `Modelo_Relatorio_Experimentos_IMD1103(1).docx` | Estrutura obrigatória do relatório. |
| `chapters-github.tar.gz` | Código do livro: `Chapter11/`, `Chapter12/`, `requirements.txt`, `setup-venv.sh`. |
| `requirements(1).txt` | Dependências preexistentes no diretório da disciplina (arquivo local chamado `requirements.txt`). |
| `setup-venv.sh` (do curso) | Script preexistente de preparação do `.venv` da disciplina. |

**Situação no diretório de trabalho do usuário:** os diretórios `Chapter11` e `Chapter12` do arquivo do livro já foram copiados para o diretório do curso, que possui ambiente Python e dependências próprios. O caminho absoluto desse diretório ainda não foi registrado.

### Código efetivamente presente no pacote do livro

```text
Chapter11/
  01_cartpole_dqn.py
  02_cartpole_reinforce.py
  03_cartpole_reinforce_baseline.py
  04_cartpole_pg.py
  05_pong_pg.py
  05_pong_pg_tune.py
  lib/common.py
Chapter12/
  01_cartpole_pg.py
  02_pong_a2c.py
  02_pong_a2c_tune.py
  03_a3c_grad.py
  lib/common.py
requirements.txt          # dependências originais do livro
setup-venv.sh             # setup original do livro
```

**Atenção:** há módulos `lib/common.py` separados em cada capítulo. Preservar a forma de importação esperada por cada script; evitar consolidá-los sem necessidade.

## 3. Experimentos e prioridades

| ID | Experimento | Código de partida | Pergunta principal | Prioridade |
|---|---|---|---|---|
| **E01** | REINFORCE × DQN no CartPole | `Chapter11/01_cartpole_dqn.py`; `02_cartpole_reinforce.py` | Como os dois métodos diferem em convergência, estabilidade e uso de interações? | **1 — primeiro a executar** |
| **E02** | REINFORCE sem × com baseline | `Chapter11/02_cartpole_reinforce.py`; `03_cartpole_reinforce_baseline.py` | Como a centralização dos retornos afeta o aprendizado? | 1 |
| **E03** | Gradientes de PG sem × com baseline | `Chapter12/01_cartpole_pg.py` (opção `--baseline`) | Que diferenças aparecem nas medidas de gradientes e recompensas? | 1 |
| **E04** | A2C no Pong | `Chapter12/02_pong_a2c.py` | O agente aprende e quais indicadores ajudam a explicar sua evolução? | 1 |
| **E05** | PG × A2C no Pong | `Chapter11/05_pong_pg.py`; `Chapter12/02_pong_a2c.py` | Como o comportamento das implementações difere diante de orçamentos comparáveis? | 2 — comparação integradora |
| **E06** | A2C em ambientes síncronos × assíncronos | `Chapter12/02_pong_a2c.py` (sem/com `--use-async`) | Qual o efeito da coleta assíncrona sobre vazão e tempo de execução? | 3 — extensão |
| **E07** | A3C com paralelismo de gradientes | `Chapter12/03_a3c_grad.py` | Que ganhos e custos surgem ao distribuir o cálculo dos gradientes? | Condicional à viabilidade |

**Núcleo do trabalho:** E01–E04. **Comparação complementar recomendada:** E05. E06 e E07 dependem do custo operacional e do funcionamento do ambiente Pong.

### E01 — REINFORCE × DQN / CartPole

- **Hipótese de trabalho:** REINFORCE terá aprendizado competitivo no CartPole, mas poderá diferir do DQN em estabilidade e eficiência de amostragem.
- **Variável investigada:** algoritmo (DQN ou REINFORCE), mantendo o ambiente e um protocolo de avaliação comum.
- **Medidas:** recompensa por episódio; média móvel de 100 episódios; passos acumulados; episódios até o critério; duração; desempenho de avaliação independente.
- **Cuidados:** os algoritmos originais têm hiperparâmetros e procedimentos de atualização distintos. A comparação não isola causalmente um único mecanismo; os hiperparâmetros originais serão mantidos na primeira reprodução.

### E02 — Efeito do baseline / CartPole

- **Hipótese:** centralizar os retornos pode reduzir a oscilação do aprendizado e melhorar a estabilidade.
- **Comparação:** scripts `02` e `03` do capítulo 11, preservando os parâmetros comuns.
- **Especificidade do código:** `03_cartpole_reinforce_baseline.py` calcula os retornos descontados de **cada episódio** e subtrai sua média **dentro do próprio episódio**. Esse mecanismo não é idêntico ao baseline por média acumulada de E03; não tratá-los como a mesma implementação.
- **Medidas:** recompensa, passos até convergência, estabilidade e dispersão entre execuções.

### E03 — Baseline e medidas dos gradientes / CartPole

- **Hipótese:** o baseline alterará a magnitude e a dispersão dos gradientes, podendo favorecer treinamento mais estável.
- **Comparação:** `Chapter12/01_cartpole_pg.py` sem parâmetro versus `--baseline`.
- **Instrumentação já presente:** `grad_l2`, `grad_max`, `grad_var`, `kl`, `entropy`, perdas e recompensas no TensorBoard.
- **Ressalva metodológica:** o `grad_var` do livro é a variância **entre os elementos de um vetor de gradientes em uma atualização**, não diretamente a variância estatística **entre estimativas do gradiente obtidas de amostras independentes**. Identificar a métrica corretamente no relatório.
- **Ressalva de implementação:** o script original encerra ao ultrapassar média móvel 195, enquanto os exemplos anteriores usam 450. Padronizar antes de uma comparação integrada.

### E04 — A2C / Pong

- **Objetivo:** reproduzir o exemplo de Actor-Critic e entender os efeitos do crítico sobre o aprendizado.
- **Elementos:** rede convolucional com corpo compartilhado e cabeças de política e valor; estimativa de vantagem; retorno n-step; bônus de entropia; clipping de gradientes.
- **Medidas:** recompensa média; vantagem; valor estimado; perdas de política, de valor e de entropia; métricas de gradientes; frames/passos; tempo e throughput.
- **Custo:** o capítulo 12 menciona treinamentos da ordem de milhões de frames, chegando a aproximadamente 8–10 milhões para bons resultados. Dimensionar orçamento e capacidade antes de executar.

### E05 — PG × A2C / Pong

- **Objetivo:** relacionar as conclusões dos capítulos 11 e 12.
- **Hipótese:** a implementação A2C tende a ser mais estável e eficaz no Pong do que a implementação PG convencional examinada pelo livro.
- **Controle:** comparar curvas sob um orçamento de interações explicitamente definido e sob a mesma métrica de avaliação.
- **Limite interpretativo:** os scripts não diferem **apenas** pela presença do crítico. Também há diferenças de configuração e possivelmente detalhes de amostragem/treino. Logo, observar vantagem do A2C **não demonstra isoladamente** que a causa foi o crítico.

### E06 — A2C síncrono × assíncrono / Pong

- **Comparação:** `SyncVectorEnv` versus `AsyncVectorEnv` com o mesmo script do capítulo 12.
- **Medidas principais:** throughput (interações e/ou frames por segundo, definidos consistentemente), tempo até determinado número de interações e recompensa alcançada nesse orçamento.
- **Controle:** manter quantidade de ambientes, arquitetura e hiperparâmetros sempre que possível.
- **Terminologia:** `--use-async` altera **a execução dos ambientes**; o gradiente continua centralizado. Não confundir com a implementação de paralelismo de gradientes do E07.

### E07 — Gradientes distribuídos / Pong

- **Base:** `Chapter12/03_a3c_grad.py`, que usa `torch.multiprocessing`, múltiplos processos e uma fila de gradientes para atualização central.
- **Objetivo:** investigar o custo-benefício do paralelismo de gradientes.
- **Cuidados de validação:** verificar compartilhamento de parâmetros, dispositivo, sincronização dos trabalhadores, contabilização global dos passos e encerramento dos processos. Não presumir comparabilidade direta com E06 sem adequar os orçamentos e a instrumentação.

## 4. Protocolo experimental comum — proposta inicial

Os valores abaixo foram **propostos, não validados experimentalmente**. Ajustes exigem registro no histórico de decisões.

| Aspecto | Proposta |
|---|---|
| Ambiente inicial | `CartPole-v1` |
| Sementes de treinamento | `42`, `123`, `2026` |
| Repetições | 3 por configuração no CartPole; no Pong, decidir após piloto de custo |
| Orçamento inicial CartPole | Até **200.000 passos de interação por execução**; reavaliável após piloto |
| Critério comparável CartPole | Média das recompensas dos **últimos 100 episódios completos ≥ 450** (exigir 100 episódios antes de declarar convergência) |
| Avaliação após treino | 20 episódios independentes por modelo; sem atualização de pesos |
| Unidade primária de progresso | Interações com o ambiente; reportar também episódios, atualizações e duração |
| Recompensa | Registrar recompensa por episódio e média móvel de 100 episódios |
| Registro | CSV para métricas; JSON para configuração/metadados; checkpoints para pesos; TensorBoard opcional/complementar |
| Hiperparâmetros iniciais | Manter os valores dos scripts do livro na reprodução inicial |

### Regras de comparação

1. **Não comparar apenas por número de episódios.** Um agente que prolonga os episódios realiza mais interações por episódio.
2. **Definir precisamente o contador:** distinguir passos do ambiente, transições coletadas, *frames* do Atari (que podem envolver *frame skip*) e atualizações do otimizador. Verificar como cada script conta seus passos antes de produzir gráficos cruzados.
3. **Separar treino e avaliação:** mesmo ambiente e regras; avaliação sem aprendizagem. Registrar se a ação da política foi amostrada ou selecionada por `argmax`. Para REINFORCE, a política estocástica de treinamento e a seleção determinística de avaliação podem produzir resultados diferentes; documentar a escolha.
4. **Não declarar convergência prematura:** a média de 100 episódios só deve sustentar o critério depois de 100 episódios completos. Quando um treinamento atingir o orçamento sem convergir, registrar como **não convergiu no orçamento**, não como fracasso definitivo do método.
5. **Usar sementes registradas e inicializações explícitas** para Python, NumPy, PyTorch e ambientes; aceitar que algumas operações paralelas podem não ser perfeitamente determinísticas.
6. **Não usar um conjunto de dados antigo como replay de algoritmos on-policy.** Respeitar a lógica de atualização dos scripts originais.
7. **Evitar tuning amplo** até reproduzir os exemplos básicos. Alterar uma variável investigada por vez quando o objetivo for atribuição de efeitos.

### Métricas mínimas para relatório

- Recompensa média e dispersão entre repetições.
- Curvas de aprendizado com o eixo horizontal claramente definido.
- Quantidade de interações e duração até o critério (quando atingido).
- Proporção de execuções que alcançaram o critério dentro do orçamento.
- Resultado da avaliação final, com média e desvio padrão, indicando política de avaliação.
- Para E03/E04: perdas, magnitude dos gradientes, entropia e métricas específicas disponíveis.
- Para E06/E07: eficiência computacional e observações de paralelização.

**Gráficos previstos:** curvas de recompensa versus passos/interações; tabelas de média/desvio padrão; gráficos de gradientes para E03; vantagens/valores/perdas para E04; throughput e tempo para E06/E07.

## 5. Infraestrutura e compatibilidade de dependências

### Situação atual (não resolvida)

- O **curso usa Python 3.12** em seu `setup-venv.sh`; o **script original do livro exige Python 3.11**.
- Foi acordado **preservar Python 3.12** e buscar compatibilidade, em vez de regredir o ambiente da disciplina.
- O arquivo de requisitos do curso usa, em geral, dependências **sem pinagem de versão**; o livro fixa versões específicas.
- O ambiente instalado atualmente **ainda não foi inventariado**; portanto, não se sabe quais versões estão em uso nem se há conflitos reais de resolução.

### Requisitos do livro (arquivo incluído no `.tar.gz`)

```text
gymnasium[atari]==0.29.1
gymnasium[classic-control]==0.29.1
gymnasium[accept-rom-license]==0.29.1
moviepy==1.0.3
numpy<2
opencv-python==4.10.0.84
torch==2.5.0
torchvision==0.20.0
pytorch-ignite==0.5.1
tensorboard==2.18.0
mypy==1.8.0
ptan==0.8.1
stable-baselines3==2.3.2
torchrl==0.6.0
ray[tune]==2.37.0
pytest
```

### Dependências do curso (lista fornecida)

`torchinfo`, `torch`, `torchvision`, `matplotlib`, `scikit-learn`, `prdc`, `torchmetrics`, `imageio`, `imageio-ffmpeg`, `gymnasium` com extras `atari`, `accept-rom-license`, `classic-control`, `opencv-python`, `moviepy`, `gymnasium-robotics`, `pyvirtualdisplay`, `gdown`, `tensorflow`, `nbconvert[webpdf]`, `numpy`, `ipykernel`, `pandas`, `tdqm`.

O item **`tdqm`** deve ser verificado: pode ser um engano em relação a **`tqdm`**. Não alterar silenciosamente sem conferir a dependência pretendida.

### Pontos de atenção técnicos

- Verificar compatibilidade entre **Gymnasium / Gymnasium Robotics / ALE / Shimmy / wrappers Atari** antes de executar Pong. O nome de ambiente `PongNoFrameskip-v4` e os wrappers do livro podem exigir adaptação para versões novas.
- Verificar compatibilidade entre **NumPy / OpenCV / PyTorch** e demais bibliotecas já utilizadas no curso.
- `Chapter11/03_cartpole_reinforce_baseline.py` importa **`tensorboardX`**, diferentemente dos demais exemplos, que usam `torch.utils.tensorboard`. Decidir entre adicionar a dependência ou padronizar o import, registrando a alteração.
- Os exemplos usam **PTAN** e, para registros do PyTorch, **TensorBoard**; checar sua disponibilidade e a resolução de dependências no ambiente efetivo.
- Antes dos exemplos Pong, verificar o funcionamento do backend **MPS** do Mac com Apple Silicon e o suporte à execução multiprocesso; não presumir que GPU será mais rápida ou compatível com todos os scripts.
- O `setup-venv.sh` do curso contém **`rm -rf .venv`**. **Não executar esse script antes de definir as dependências**, pois ele destrói o ambiente existente.
- O script do curso cita `recreate-venv.sh` na mensagem de erro, mas o arquivo compartilhado se chama `setup-venv.sh`.

**Estratégia acordada:** preservar primeiro o ambiente do curso; inventariar as versões instaladas; adicionar somente o que os experimentos precisam; adaptar APIs quando necessário; validar regressão dos trabalhos anteriores; não copiar cegamente a lista de versões do livro sobre a do curso.

### Próxima evidência solicitada ao usuário

No diretório do curso, **sem recriar `.venv`**:

```bash
source .venv/bin/activate
python --version
python -m pip freeze > requirements-instalados.txt
python -m pip check
```

O usuário enviará `requirements-instalados.txt` e a saída de `pip check`. **Essa verificação ainda não ocorreu neste plano.** Somente após conhecer esse inventário será proposta a consolidação das dependências.

## 6. Preparação da execução e instrumentação

**Diretriz:** preservar os scripts originais do livro na primeira reprodução, salvo correções indispensáveis de compatibilidade. Manter alterações de instrumentação pequenas, rastreáveis e preferencialmente compartilhadas entre experimentos.

Estrutura sugerida, **ainda não criada**:

```text
<diretorio-do-curso>/
  Chapter11/                    # código original do livro
  Chapter12/                    # código original do livro
  requirements.txt              # do curso, a compatibilizar
  setup-venv.sh                 # do curso, com Python 3.12
  experimentos/
    configs/
    results/
      <run_id>/
        config.json
        episodes.csv
        updates.csv             # opcional, quando aplicável
        evaluation.json
        checkpoint.pt
    plots/
    reports/
```

Esta é **uma sugestão de organização**, não uma imposição de reorganização dos arquivos já existentes.

### Contrato mínimo sugerido para cada execução

- `run_id` único, por exemplo `E01_reinforce_seed42_YYYYMMDD-HHMMSS`.
- `config.json`: ID, algoritmo, script e revisão Git, versão Python/bibliotecas, dispositivo, ambiente e wrappers, sementes, hiperparâmetros, limite de passos, regra de avaliação e duração.
- `episodes.csv`: `run_id`, `episode`, `env_steps_total`, `reward`, `reward_mean_100`, `elapsed_seconds`.
- `updates.csv` quando pertinente: número de atualização, passos acumulados, perdas, entropia, gradientes e/ou estimativas de valor. Não preencher artificialmente métricas indisponíveis.
- `evaluation.json`: número de episódios, sementes de avaliação, tipo de seleção de ações, recompensas, média, desvio padrão e taxa de convergência, se cabível.
- `checkpoint.pt`: pesos do modelo e demais estados exigidos para retomar a execução, se esse recurso for implementado.

### Critérios para considerar a instrumentação pronta

- [ ] Configuração e semente aparecem no resultado salvo.
- [ ] O contador de interações é compreendido e testado.
- [ ] CSV e metadados são gerados em locais próprios, sem sobrescrever outras execuções.
- [ ] Os critérios de parada podem ser configurados sem alterar o algoritmo.
- [ ] Há avaliação independente do treino.
- [ ] Os exemplos originais continuam reconhecíveis e as alterações estão documentadas.

## 7. Ordem de trabalho e marcos

### Fase 0 — Compatibilidade do ambiente (em andamento)

1. Obter inventário do `.venv` existente e `pip check`.
2. Avaliar conflitos reais e selecionar o menor conjunto de mudanças necessárias.
3. Ajustar `requirements.txt` do curso **somente depois** de validar a proposta.
4. Validar imports, criação de `CartPole-v1` e funcionamento dos scripts mínimos.
5. Garantir que os experimentos anteriores do curso continuam funcionais.

**Entrega:** ambiente Python 3.12 validado para o CartPole, com mudanças registradas. A validação do Atari pode ocorrer separadamente antes do E04.

### Fase 1 — Instrumentação mínima do E01

1. Validar execução breve dos dois scripts originais.
2. Acrescentar configuração da semente, limite de interações e critério de parada.
3. Criar registros de episódios e configuração.
4. Criar avaliação final independente e persistência de resultados.
5. Validar geração de tabelas e gráficos de uma execução curta.

**Entrega:** experimento E01 reexecutável, sem conclusão experimental ainda.

### Fase 2 — Execução e análise do E01

1. Piloto DQN e REINFORCE para verificar custo e estabilidade da instrumentação.
2. Validar ou ajustar, **explicitamente**, o orçamento de 200 mil passos.
3. Executar as sementes propostas, controlando condições comuns.
4. Analisar recompensas, uso de interações, duração e avaliação final.
5. Registrar a subseção E01 de resultados do relatório.

**Entrega:** E01 documentado com dados, gráficos, limitações e observações.

### Fase 3 — E02 e E03 (baseline / CartPole)

1. Executar variantes do REINFORCE com e sem baseline (E02).
2. Executar `Chapter12/01_cartpole_pg.py` com e sem `--baseline` (E03).
3. Comparar os resultados; explicar que os tipos de baseline e os protocolos dos scripts diferem.
4. Registrar efeito observado nas medidas de gradiente, sem alegar prova de causalidade além do desenho feito.

### Fase 4 — Validação Atari e E04 (A2C / Pong)

1. Resolver compatibilidade do ambiente Pong e dos wrappers com o Python 3.12.
2. Medir throughput, memória e custo de uma execução curta.
3. Estabelecer orçamento de frames/tempo adequado à máquina.
4. Executar A2C com checkpoint e métricas apropriadas.
5. Documentar até onde o treinamento chegou, inclusive ausência de convergência.

### Fase 5 — E05 e extensões opcionais

1. E05: comparar PG e A2C no Pong por orçamento comum, declarando diferenças de implementação.
2. E06: comparar A2C síncrono e assíncrono com métricas computacionais.
3. E07: executar paralelismo de gradientes **se** houver tempo, capacidade e validação técnica suficientes.

### Fase 6 — Consolidação do relatório

Revisar evidências, gráficos, tabelas e interpretação conjunta. Não fabricar dados para preencher seções; experimentos incompletos devem ser reportados como tal.

## 8. Estrutura do relatório exigido

Respeitar a organização do arquivo `Modelo_Relatorio_Experimentos_IMD1103(1).docx`:

1. **Introdução:** contexto, objetivo, problema e hipóteses.
2. **Fundamentação Teórica:** algoritmos e ambientes (estados, ações, recompensas, término e dificuldades).
3. **Metodologia Experimental:** configurações, sementes, repetições, parada, ferramentas, treino, avaliação e visualizações.
4. **Resultados:** gráficos, tabelas e descrição qualitativa; subseções por experimento.
5. **Discussão:** confronto das hipóteses com resultados, explicação teórica, limitações e melhorias.
6. **Conclusões:** síntese do que foi observado, limites e possíveis investigações seguintes.

O documento final deve ser **um relatório integrado**, não sete relatórios isolados. É aceitável omitir E06/E07 ou registrá-los como trabalho futuro se não forem executados.

## 9. Decisões, pendências e perguntas em aberto

### Decisões já tomadas

- Usar o **código do livro** como base dos experimentos.
- Priorizar o **CartPole e o E01**, avançando gradualmente.
- Manter **Python 3.12**, buscando compatibilidade com o projeto do curso.
- **Não destruir ou substituir o ambiente do curso** antes de inventariá-lo.
- Manter a estrutura de seis seções do relatório exigido.
- Usar E01–E04 como núcleo, E05 como comparação integradora, E06/E07 como extensões condicionais.
- **Discutir e preparar** primeiro; não iniciar treinamentos antes da validação do ambiente e da instrumentação mínima.

### Pendências imediatas

- [ ] Receber `requirements-instalados.txt` e resultado de `python -m pip check`.
- [ ] Resolver compatibilização das dependências para Python 3.12.
- [ ] Validar uma execução curta de `CartPole-v1` com o código do livro.
- [ ] Implementar instrumentação mínima e execução piloto de E01.

### Decisões que continuam abertas

- Versões finais das dependências; necessidade de ambientes separados caso compatibilidade única seja impraticável.
- Forma concreta de instrumentar os scripts sem duplicar código ou modificar a lógica de aprendizado.
- Tratamento de episódios truncados por limite do ambiente, especificamente para cálculo de retornos e avaliação.
- Se o orçamento inicial de 200 mil passos é adequado aos dois agentes do E01.
- Política de ação na avaliação final (determinística, estocástica ou ambas, conforme a pergunta).
- Repetições e orçamento do Pong depois do piloto de desempenho.
- Viabilidade e custo do A3C em macOS/Apple Silicon.

## 10. Protocolo para transferência a outros agentes de IA

Ao delegar uma tarefa, fornecer **este documento** e, quando houver código envolvido, os arquivos relevantes do repositório. A tarefa deve explicitar:

1. **Escopo e ID:** por exemplo, “Fase 0 — compatibilidade” ou “E01 — instrumentação”.
2. **Arquivos autorizados:** caminhos que o agente pode ler e alterar.
3. **Objetivo verificável:** resultado que deve ser apresentado, sem adiantar etapas.
4. **Restrições:** preservar o Python 3.12, experimentos anteriores, scripts de referência e metodologia registrada.
5. **Evidências esperadas:** diff de código, comandos executados, versões, logs, tabelas ou métricas, conforme o trabalho.
6. **Atualização deste plano:** sugerir alterações nas seções pertinentes, sem reescrever retrospectivamente decisões ou afirmar execução não ocorrida.

**Regras operacionais para agentes:**

- Não presumir que dependências, ROMs Atari, GPU/MPS ou experimentos já estejam funcionais.
- Não executar `setup-venv.sh` nem apagar `.venv` sem autorização explícita.
- Não promover uma **hipótese** a **resultado** sem dados.
- Não alterar múltiplos hiperparâmetros para estudar um único efeito sem registrar a limitação.
- Não trocar o ambiente de execução, algoritmo ou métricas sem registrar a mudança.
- Não iniciar as fases E02–E07 automaticamente só porque E01 foi configurado.
- Ao concluir uma tarefa, relatar: **alterações feitas**, **verificações realizadas**, **resultados**, **limitações** e **próxima pendência**.

### Ponto exato de retomada

**Retomar pela Fase 0:** aguardar/avaliar o inventário de pacotes realmente instalados no `.venv` do curso e a saída de `pip check`. Em seguida, propor a compatibilização mínima de dependências do livro com Python 3.12, sem executar ainda `setup-venv.sh` nem iniciar os treinamentos.

---

**Controle de alterações**

| Data | Registro |
|---|---|
| 2026-11-22 | Versão inicial: objetivos das aulas, sequência E01–E07, protocolo preliminar, requisitos conhecidos, pendências de compatibilidade e instruções para agentes. |
