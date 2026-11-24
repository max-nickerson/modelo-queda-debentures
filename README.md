# Modelo de risco de queda de debêntures

Um arquivo Python (`modelo_queda_debentures.py`), só com pandas e numpy, que ordena as debêntures pela chance de **cair forte nos próximos 3 meses**. Este README é o contexto completo para quem for rodar ou adaptar o modelo, pessoa ou assistente de IA: o que ele faz, quais dados usa, em que formato, o que já foi testado e as armadilhas.

---

## 1. O que o modelo prevê (e o que não prevê)

**Alvo.** Para cada debênture, no fim de cada mês, o modelo dá a probabilidade de o retorno dos 3 meses seguintes ficar **5% ou mais abaixo da mediana das debêntures parecidas**. Parecidas = mesmo mês, mesmo indexador (DI ou IPCA) e mesma faixa de duration (<2, 2–4, 4–6, >6 anos). Isso é uma "queda forte"; a base é de ~2,5% das debênture-meses.

**Uso.** É um **ranking de risco**, não um preditor de preço. Serve para decidir o que revisar e onde não aumentar posição. Em crédito, evitar as poucas que quebram vale mais que achar as que sobem.

**Não prevê:**
- **Subida / fechamento de spread** de debêntures saudáveis: o efeito é praticamente zero.
- **Choques súbitos** sem sinal prévio no preço, como fraude contábil (Americanas jan/23) ou um grupo inteiro caindo de uma vez.
- **Antes de 2022:** em 2019–2021 o sinal não funcionava (AUC 0,34–0,55). Só confie de 2022 em diante.

## 2. Como funciona

São dois modelos (regressões logísticas). Cada variável entra como **posição relativa dentro do mês** (rank de 0 a 1 entre todas as debêntures daquele mês).

| Debênture… | Variáveis | Qualidade (AUC fora da amostra, 2022–26) |
|---|---|---|
| **com ação listada** (do emissor ou do controlador) | tendência da ação em 6 meses + lag da debênture + desconto + desconto² | ~0,84 |
| **sem ação listada** | lag + desconto + desconto² + rating/outlook/rebaixamento recente + alavancagem líquida (+ nº de assembleias de debenturistas) | ~0,74 |

AUC: 0,5 = chute, 1 = perfeito.

**Variáveis:**
- **Tendência da ação (`trend6`):** soma dos retornos diários da ação nos últimos 126 dias úteis, **sem a parte explicada pelos fatores de mercado** e **sem os dias de salto** (resíduo > 3 desvios). Os fatores são Ibovespa, dólar, DI 5 anos, Brent, minério, soja, açúcar e cobre. Os betas são estimados nos 252 dias **antes** da janela. O intercepto (drift do ano anterior) **não** é subtraído.
- **Lag (`lag3`):** retorno da debênture nos últimos 3 meses menos a mediana das parecidas.
- **Desconto (`disc`):** preço da debênture (% do par) menos a mediana das parecidas.
- **Alavancagem líquida:** (dívida − caixa) ÷ ativo total, do último balanço já publicado.
- **Rating:** nível numérico (1 = AAA, maior = pior), outlook, watch, e se houve rebaixamento nos últimos 6 meses.

**Treino:** usa todos os meses com resultado já conhecido (até 3 meses antes do último). A penalidade L2 é equivalente a `C = 0,5` do sklearn.

**Gatilho de curto prazo** (precisa de dados diários). Dentro do top 10%, a taxa de queda forte vai de ~6% sem gatilho para ~20% com gatilho, uns 25 dias antes. Dispara quando há:
- um negócio de R$ 1 mi ou mais pelo menos 1 ponto de % do par abaixo da marca do dia anterior, ou
- a marca caiu 1,5% ou mais em ~20 dias úteis.

**Faixas na saída:** VERMELHO = top 10% do mês; AMARELO = top 10–30%.

**Termômetro de mercado (o "quando").** É impresso no início e salvo em `saida_modelo/termometro_AAAA-MM.txt`.
- **Base:** a taxa de quedas fortes dos últimos 3 meses com resultado conhecido, comparada com a taxa normal. As quedas vêm em sequência: correlação de +0,49 com a taxa dos 3 meses seguintes, o melhor indicador encontrado. Acima de 1,25× o normal = "fase ruim: agir sobre a lista vermelha".
- **Reforços fracos:** amplitude (% de debêntures DI caindo ≥ 0,25 pt de % do par no mês, acima do próprio percentil 80 histórico) e emissão primária fraca (precisa de `data_emissao`/`volume_emissao` no cadastro).
- **Testado e reprovado:** contar alarmes de dólar, juros, Ibov abaixo da média de 200 dias e resgates. Dólar e juros subindo vieram antes de *menos* quedas em 2020–26.

**Valor para a carteira** (2022–26). Marcando para revisão a fatia mais arriscada:

| Fatia marcada | R$ perdido em quedas fortes que é pego |
|---|---|
| 10% da carteira | ~53% |
| 20% | ~69% |
| 30% | ~78% |

## 3. Como rodar

```bash
pip install pandas numpy        # openpyxl só se for ler .xlsx
```
```bash
python modelo_queda_debentures.py --modelos
```
Cria `dados_modelo/` com um CSV vazio por fonte (só o cabeçalho) e um `LEIA-ME.txt`.
```bash
python modelo_queda_debentures.py --backtest
```
Qualidade ano a ano, só com dados do passado (walk-forward com 4 meses de intervalo), e depois a lista.
```bash
python modelo_queda_debentures.py
```
Gera `saida_modelo/lista_risco_AAAA-MM.csv` (`;` como separador e vírgula decimal, abre direto no Excel) e `saida_modelo/termometro_AAAA-MM.txt`.

**Formato dos arquivos:**
- CSV com `;` ou `,`, ou `.xlsx`.
- Números com vírgula ou ponto decimal.
- Datas `dd/mm/aaaa` ou `aaaa-mm-dd`.
- Nomes de coluna em minúsculas, exatamente como abaixo.

## 4. Dados de entrada (pasta `dados_modelo/`)

| Arquivo | Obrigatório | Colunas | Frequência | Observações |
|---|---|---|---|---|
| `precos_debentures` | **sim** | `data; codigo; pct_par` (+ opc. `duration_anos; data_publicacao`) | diário (ANBIMA indicativa) ou mensal | `pct_par` = PU ÷ PU par × 100. **Desde 2019**, incluindo as debêntures que deram default, foram reestruturadas, resgatadas ou venceram. `data_publicacao` só se o preço sai com atraso (ex.: marcas da CVM) |
| `cadastro_debentures` | **sim** | `codigo; cnpj_emissor; indexador; vencimento` (+ opc. `emissor; data_emissao; volume_emissao`) | atualizar mensalmente | Uma linha por debênture, todas. `indexador` contendo "DI" (CDI+, %CDI) ou "IPCA"; os outros são ignorados. `data_emissao`/`volume_emissao` (R$) só servem para o termômetro |
| `mapa_emissor_acao` | não, mas é o que mais pesa | `cnpj_raiz; ticker` | quando muda o controle | `cnpj_raiz` = 8 primeiros dígitos do CNPJ do emissor. Ticker do **próprio emissor ou do controlador listado** (SPE → holding, subsidiária → controladora). Pode ter mais de um ticker por emissor (faz a média) |
| `precos_acoes` | não | `data; ticker; fechamento` | diário | Fechamento **ajustado** por dividendos e desdobramentos. Pelo menos 18 meses antes do primeiro mês analisado |
| `fatores` | não | `data; ibov` (+ opc. `usdbrl; di5y; brent; minerio; soja; acucar; cobre`) | diário | Níveis (pontos, preço, taxa). `di5y` = taxa do DI futuro 5 anos em % a.a. Sem esse arquivo, usa o retorno bruto da ação (perde ~0,03 de AUC) |
| `ratings` | não | `cnpj_raiz; data; rating_num` (+ opc. `outlook; watch`) | uma linha por mudança | `data` = quando a mudança ficou conhecida. `rating_num`: 1 = AAA, 2 = AA+, …; `outlook` −1/0/1; `watch` 1 = observação negativa. Rating interno serve |
| `balancos` | não | `cnpj_raiz; data_entrega; divida; caixa; ativo_total` | trimestral | `divida` = empréstimos e financiamentos + debêntures, curto e longo prazo. `data_entrega` = data em que o balanço ficou público. Fonte grátis: CVM ITR/DFP (ver seção 6) |
| `assembleias` | não | `cnpj_raiz; data_entrega` | diário | Uma linha por documento de assembleia de debenturistas (convocação ou ata), do sistema IPE da CVM |
| `negocios` | não | `data; codigo; pct_par; volume_brl` | diário / intradiário | Negócios executados (B3, SND e mesa). **Preço em % do par**: se vier em taxa, converter antes |
| `posicoes` | não | `fundo; codigo; valor_brl` | a data atual | Só para mostrar quanto cada fundo tem em cada nome da lista |

## 5. Regras que não podem ser quebradas

1. **Sem informação do futuro.** Toda variável só entra a partir da data em que ficou pública:
   - balanço: data de entrega;
   - rating: data da mudança;
   - preço publicado com atraso: `data_publicacao`.
2. **Manter as debêntures que saíram do mercado** (default, reestruturação, resgate, vencimento) no histórico. Sem elas, o modelo não vê as quedas que deveria aprender.
3. **Retornos são % do PU par** (PU ÷ PU par). Juros e amortização não são somados, porque PU e PU par caem juntos no pagamento.
4. **Marca da ANBIMA serve para medir risco, não para simular operação.** Para medir resultado de operar (comprar ou vender), use negócios executados.
5. **Testar sempre walk-forward:** treinar só no passado, com intervalo de 4 meses porque o alvo é de 3 meses. Nunca ajustar cortes olhando o ano de teste.

## 6. Mapeamento de debêntures e outras fontes que ajudam

- **Emissor → ação (o mais importante).** Cada debênture que ganha uma ação passa do modelo de ~0,74 para o de ~0,84. Fontes:
  - cadastro de companhias da CVM;
  - Formulário de Referência (FRE) da CVM, com os acionistas controladores (`dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FRE/`);
  - base interna de grupo econômico, se houver.

  Cuidado com controladores "hub" (governo, fundos de pensão, BNDESPAR): ligam empresas que não têm nada a ver.
- **Balanços grátis:** `dados.cvm.gov.br/dados/CIA_ABERTA/DOC/ITR/` e `/DFP/`. Em cada zip:
  - arquivo índice com `DT_RECEB` (use a **primeira versão** entregue de cada balanço);
  - `BPP_con`: contas `2.01.04` + `2.02.01` = dívida;
  - `BPA_con`: contas `1.01.01` + `1.01.02` = caixa, conta `1` = ativo total.

  Use o consolidado e, se faltar, o individual. Emissoras fechadas registradas na CVM também entregam esses arquivos, o que cobre ~40% das debêntures sem ação.
- **Assembleias:** `dados.cvm.gov.br/dados/CIA_ABERTA/DOC/IPE/`, documentos de assembleia de debenturistas.
- **Ações e fatores:** B3 COTAHIST (grátis, mas sem ajuste: ajustar por proventos), Banco Central (PTAX), B3 (DI futuro), ou Bloomberg.
- **Rating interno do banco:** nos dados públicos só ~12% das debêntures sem ação têm rating, e o rating foi o bloco que melhorou o modelo das sem ação em todos os anos.
- **Ainda não testado por falta de dados (promissor):**
  - preço de **eurobonds** das emissoras fechadas, que pode fazer o papel da ação;
  - **limites de covenant** nas escrituras, para medir a distância até o default técnico;
  - **taxa de aluguel** de ações (a B3 só dá ~21 dias de histórico grátis).

## 7. O que já foi testado e NÃO ajuda (não refazer)

- Parte do movimento da ação explicada por Ibov, dólar, juros ou commodities, e os betas a esses fatores.
- Exposição pelo balanço: dívida em moeda estrangeira, "choque de dólar" e derivativos.
- Limpar também o movimento do setor da ação (piora; a queda do setor também é risco).
- Qualidade/formato da tendência (efficiency ratio, R²), "Janus" (dias de alta vs. queda do Ibov), saltos de um dia, reversão do último mês, volume anormal.
- Opções (vol implícita, skew): repetem a informação da ação e não sobem antes da queda.
- Muro de vencimentos: efeito invertido.
- Fundos vendendo (carteiras da CVM chegam com até 90 dias de atraso).
- Debêntures irmãs do mesmo emissor.
- Grafo de grupo econômico: o grupo cai junto, sem um membro avisar antes.
- Modelo com dezenas de variáveis ou composto de ranks com pesos iguais: pior que o modelo enxuto fora da amostra.

## 8. Testes de ponta a ponta deste arquivo

O arquivo foi rodado com dados reais convertidos para o formato da seção 4.

**Teste 1: marcas mensais dos fundos na CVM, 2019–2026, sem viés de sobrevivência**, com ações, fatores, ratings, balanços e assembleias.

| Ano | AUC com ação | AUC sem ação | Queda no top 10% vs geral (com ação) |
|---|---|---|---|
| 2022 | 0,78 | 0,77 | 5,7% vs 1,1% |
| 2023 | 0,69 | 0,67 | 5,5% vs 2,2% |
| 2024 | 0,86 | 0,76 | 9,2% vs 1,8% |
| 2025 | 0,88 | 0,75 | 21,3% vs 3,2% |
| 2026 | 0,94 | 0,70 | 24,5% vs 3,4% |

As assembleias melhoram o modelo sem ação em todos os anos (+0,01 a +0,04).

**Teste 2: marcas ANBIMA diárias + negócios grandes (≥ R$ 1 mi), 2024–2026.**
- O AUC sobe para 0,94–0,97 com ação e 0,78–0,82 sem ação. Esse número é otimista: o histórico ANBIMA usado só tinha as debêntures que ainda existem (viés de sobrevivência), e o alvo e as variáveis vêm da mesma marca.
- **O gatilho funciona em todos os anos.** Dentro do top 10% fora da amostra, a taxa de queda forte é de 24,5% com gatilho contra 12,9% sem (2024: 13,7% / 5,8%; 2025: 32,3% / 14,3%; 2026: 23,6% / 17,6%).
- Num mês ruim o gatilho dispara para muitos nomes (333 de ~1.200 em out/26). Use-o para ordenar dentro do vermelho, não sozinho.

## 9. Checagens de robustez já feitas

- **Placebo:** com a ação de outro emissor sorteada no mesmo mês, o AUC cai para 0,50. O sinal é real.
- **Não é só a marca alcançando o preço:** a ação prevê igualmente a queda medida 1 e 2 meses depois (AUC ~0,78).
- **Sobrevivência:** as debêntures vermelhas não somem das carteiras mais que as outras.
- **Margem de erro por ano** (IC 90%, reamostrando emissores): 2022 0,66–0,79; 2023 0,71–0,84; 2024 0,81–0,91; 2025 0,77–0,91; 2026 0,86–0,93.

## 10. Estrutura do código (para adaptar)

| Função | O que faz |
|---|---|
| `le()` | lê e valida cada arquivo, converte números e datas |
| `painel()` | marca de fim de mês, grupos de parecidas, alvo (`rx3`, `queda`), `lag3`, `disc` |
| `tendencia()` | tendência própria da ação por emissor e mês |
| `emissor()` | rating, alavancagem e assembleias, sempre com a última informação já pública |
| `variaveis()` | ranks dentro do mês e escolha das variáveis de cada modelo |
| `ajusta()` / `prob()` / `auc()` | regressão logística em numpy e métrica |
| `backtest()` | qualidade ano a ano, walk-forward |
| `gatilho()` | gatilho de curto prazo com negócios e marcas diárias |
| `avalia_gatilho()` | backtest do gatilho dentro do top 10% (roda com `--backtest`) |
| `termometro()` | fase do mercado: quedas recentes, amplitude e emissão |
| `lista()` | lista do último mês, com faixas, gatilho e posições |

As constantes no topo do arquivo podem ser ajustadas: `QUEDA`, `FAIXAS_DURATION` e os `GATILHO_*`.

## 11. Pontos em aberto ao adaptar

1. Se o histórico ANBIMA não tiver as debêntures que saíram do mercado, complemente com as marcas mensais dos fundos na CVM (CDA), que têm `data_publicacao`.
2. Se os negócios vierem em taxa, converta para % do par antes (taxa → PU pelo fluxo da escritura ÷ PU par).
3. A escala do rating interno precisa ser convertida para números em que maior = pior.
4. Sem `duration_anos`, o código aproxima a duration por 0,75 × prazo até o vencimento. Isso piora um pouco os grupos de parecidas.
