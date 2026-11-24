"""
MODELO DE RISCO DE QUEDA DE DEBÊNTURES

O que prevê: a chance de cada debênture cair mais de 5% além das debêntures parecidas (mesmo indexador e mesma faixa de
duration) nos próximos 3 meses. É um RANKING de risco: serve para dizer o que revisar / não aumentar.

Uso (só precisa de pandas e numpy):
  python modelo_queda_debentures.py --modelos    cria dados_modelo/ com os CSVs vazios (só cabeçalho) e um LEIA-ME
  python modelo_queda_debentures.py --backtest   mede a qualidade ano a ano, só com dados do passado (walk-forward)
  python modelo_queda_debentures.py              gera saida_modelo/lista_risco_AAAA-MM.csv do último mês

Os arquivos vão em dados_modelo/ (CSV com ; ou , ou .xlsx; números com vírgula ou ponto; datas dd/mm/aaaa ou aaaa-mm-dd).
Só os dois primeiros são obrigatórios. Cada um que você acrescenta melhora o modelo (ver ARQUIVOS abaixo).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
DADOS, SAIDA = AQUI / 'dados_modelo', AQUI / 'saida_modelo'

# nome: (obrigatório, colunas obrigatórias, colunas opcionais, descrição)
ARQUIVOS = {
    'precos_debentures': (True, ['data', 'codigo', 'pct_par'], ['duration_anos', 'data_publicacao'],
                          'Preço de cada debênture em % do PU par (PU / PU par x 100). Diário (ANBIMA indicativa) ou mensal. '
                          'Desde 2019, INCLUINDO debêntures que deram default, foram reestruturadas ou resgatadas. '
                          'duration_anos: da ANBIMA (se faltar, uso o prazo até o vencimento). '
                          'data_publicacao: só se o preço for publicado com atraso (ex.: marcas da CVM).'),
    'cadastro_debentures': (True, ['codigo', 'cnpj_emissor', 'indexador', 'vencimento'], ['emissor', 'data_emissao', 'volume_emissao'],
                            'Uma linha por debênture (todas, inclusive vencidas). indexador: DI (CDI+ ou %CDI) ou IPCA; '
                            'outros indexadores são ignorados. data_emissao e volume_emissao (R$): opcionais, usados no '
                            'termômetro de mercado (emissão primária fraca).'),
    'mapa_emissor_acao': (False, ['cnpj_raiz', 'ticker'], [],
                          'Emissor (8 primeiros dígitos do CNPJ) -> ação listada do PRÓPRIO emissor ou do CONTROLADOR '
                          '(ex.: SPE -> holding listada). Sem este arquivo, todas são tratadas como "sem ação".'),
    'precos_acoes': (False, ['data', 'ticker', 'fechamento'], [],
                     'Fechamento diário AJUSTADO por dividendos e desdobramentos. Pelo menos 18 meses antes do início.'),
    'fatores': (False, ['data', 'ibov'], ['usdbrl', 'di5y', 'brent', 'minerio', 'soja', 'acucar', 'cobre'],
                'Diário. ibov e commodities em pontos/preço; usdbrl = PTAX; di5y = taxa do DI futuro 5 anos em % a.a. '
                'Serve para tirar da ação a parte que veio do mercado. Sem ele, uso o retorno bruto da ação (perde pouco).'),
    'ratings': (False, ['cnpj_raiz', 'data', 'rating_num'], ['outlook', 'watch'],
                'Uma linha por mudança de rating (agência ou rating interno). rating_num: 1 = AAA, 2 = AA+, ... (maior = pior). '
                'outlook: -1 negativo, 0 estável, 1 positivo. watch: 1 em observação negativa.'),
    'balancos': (False, ['cnpj_raiz', 'data_entrega', 'divida', 'caixa', 'ativo_total'], [],
                 'Trimestral (ITR/DFP da CVM ou base interna). divida = empréstimos + debêntures (curto + longo prazo). '
                 'data_entrega = quando o balanço ficou público.'),
    'assembleias': (False, ['cnpj_raiz', 'data_entrega'], [],
                    'Uma linha por documento de assembleia de debenturistas (convocação ou ata), IPE da CVM.'),
    'negocios': (False, ['data', 'codigo', 'pct_par', 'volume_brl'], [],
                 'Negócios executados (B3, SND e mesa): preço em % do PU par e volume em R$. Para o gatilho de curto prazo.'),
    'posicoes': (False, ['fundo', 'codigo', 'valor_brl'], [],
                 'Posição atual dos fundos (R$ por debênture). Só para mostrar quanto temos em cada nome da lista.'),
}
QUEDA = -0.05             # queda forte = retorno 3 meses 5% abaixo da mediana das parecidas
FAIXAS_DURATION = [-np.inf, 2, 4, 6, np.inf]
GATILHO_NEGOCIO_PTS = 1.0  # negócio >= R$ 1 mi pelo menos 1 ponto de % do par abaixo da marca do dia anterior
GATILHO_VOLUME = 1e6
GATILHO_MARCA_20D = -0.015  # marca caiu 1,5% ou mais em ~20 dias úteis


# ------------------------------------------------------------------ leitura
def num(s):
    """Converte '1.234,56' / '1234.56' / 1234.56 em número."""
    if s.dtype != object:
        return pd.to_numeric(s, errors='coerce')
    s = s.astype(str).str.strip()
    s = s.where(~s.str.contains(','), s.str.replace('.', '', regex=False).str.replace(',', '.', regex=False))
    return pd.to_numeric(s, errors='coerce')


def data(s):
    s = s.astype(str).str.strip()
    iso = s.str.match(r'^\d{4}-\d{2}-\d{2}')
    return pd.to_datetime(s.where(iso), errors='coerce').fillna(pd.to_datetime(s.where(~iso), dayfirst=True, errors='coerce'))


def raiz(s):
    d = s.astype(str).str.replace(r'\D', '', regex=True)
    return np.where(d.str.len() <= 8, d.str.zfill(8), d.str.zfill(14).str[:8])


def le(nome):
    obrig, cols, opc, _ = ARQUIVOS[nome]
    f = next((DADOS / f'{nome}.{e}' for e in ('csv', 'xlsx') if (DADOS / f'{nome}.{e}').exists()), None)
    if f is None:
        if obrig:
            raise SystemExit(f'Falta o arquivo obrigatório dados_modelo/{nome}.csv (rode com --modelos para ver o formato).')
        return None
    x = pd.read_excel(f, dtype=str) if f.suffix == '.xlsx' else pd.read_csv(f, sep=None, engine='python', dtype=str)
    x.columns = x.columns.str.strip().str.lower()
    falta = [c for c in cols if c not in x]
    if falta:
        raise SystemExit(f'{f.name}: faltam as colunas {falta}')
    if x.empty:
        if obrig:
            raise SystemExit(f'{f.name} está vazio')
        return None
    for c in x:
        if c.startswith('data') or c == 'vencimento':
            x[c] = data(x[c])
        elif c not in ('codigo', 'ticker', 'cnpj_emissor', 'cnpj_raiz', 'indexador', 'fundo', 'emissor'):
            x[c] = num(x[c])
    if 'codigo' in x:
        x['codigo'] = x.codigo.str.strip().str.upper()
    if 'ticker' in x:
        x['ticker'] = x.ticker.str.strip().str.upper()
    if 'cnpj_raiz' in x:
        x['cnpj_raiz'] = raiz(x.cnpj_raiz)
    print(f'  {f.name}: {len(x):,} linhas')
    return x


# ------------------------------------------------------------------ painel mensal: alvo e variáveis da debênture
def painel(D):
    px = D['precos_debentures'].dropna(subset=['data', 'pct_par'])
    px = px[px.pct_par.between(5, 200)].copy()
    px['m'] = px.data.dt.to_period('M')
    diario = px.groupby(['codigo', 'm']).size().median() > 3
    px = px.sort_values('data').groupby(['codigo', 'm'], as_index=False).last()        # marca do fim do mês
    cad = D['cadastro_debentures'].drop_duplicates('codigo').copy()
    cad['root'] = raiz(cad.cnpj_emissor)
    ix = cad.indexador.str.upper().str.replace('%', '', regex=False)
    cad['ix'] = np.where(ix.str.contains('IPCA'), 'IPCA', np.where(ix.str.contains('DI'), 'DI', None))
    P = px.merge(cad[['codigo', 'root', 'ix', 'vencimento'] + (['emissor'] if 'emissor' in cad else [])], on='codigo')
    P = P[P.ix.notna()].copy()
    P['t'] = P.m.dt.to_timestamp(how='end').dt.normalize()
    anos = (P.vencimento - P.t).dt.days / 365.25
    P['dur'] = P.duration_anos if 'duration_anos' in P and P.duration_anos.notna().any() else 0.75 * anos
    P['db'] = pd.cut(P.dur.fillna(anos), FAIXAS_DURATION, labels=False)
    s = P.set_index(['codigo', 'm']).pct_par

    def rel(col):           # menos a mediana das parecidas: mesmo mês, indexador e faixa de duration (só com >= 5 parecidas)
        gm = P.groupby(['m', 'ix', 'db'])[col]
        return (P[col] - gm.transform('median')).where(gm.transform('count') >= 5)
    P['ret3'] = s.reindex(list(zip(P.codigo, P.m + 3))).values / P.pct_par - 1
    P['rx3'] = rel('ret3').clip(-0.95, 0.5)
    P['queda'] = (P.rx3 < QUEDA).astype(float).where(P.rx3.notna())
    P['past3'] = P.pct_par / s.reindex(list(zip(P.codigo, P.m - 3))).values - 1
    P['lag3'], P['disc'] = rel('past3'), rel('pct_par')
    if 'data_publicacao' in P and P.data_publicacao.notna().any():      # preço publicado com atraso: só usa o já publicado
        K = P[['codigo', 'data_publicacao', 'lag3', 'disc']].dropna(subset=['data_publicacao']).sort_values('data_publicacao')
        P = pd.merge_asof(P.drop(columns=['lag3', 'disc']).sort_values('t'), K, left_on='t', right_on='data_publicacao',
                          by='codigo', direction='backward', tolerance=pd.Timedelta(days=120))
    return P, diario


# ------------------------------------------------------------------ tendência da ação (parte própria da empresa, sem saltos)
def tendencia(D, meses):
    A, mapa = D.get('precos_acoes'), D.get('mapa_emissor_acao')
    if A is None or mapa is None:
        return pd.DataFrame(columns=['root', 't', 'trend6'])
    A = A.dropna(subset=['data', 'fechamento']).sort_values('data')
    A['r'] = np.log(A.fechamento).groupby(A.ticker).diff().clip(-1.6, 0.7)
    A = A.merge(mapa.rename(columns={'cnpj_raiz': 'root'}), on='ticker').groupby(['root', 'data']).r.mean()
    F = D.get('fatores')
    if F is not None:
        F = F.dropna(subset=['data']).sort_values('data').set_index('data')
        fc = [c for c in ['ibov', 'usdbrl', 'brent', 'minerio', 'soja', 'acucar', 'cobre'] if c in F]
        X = np.log(F[fc]).diff()
        if 'di5y' in F:
            X['di5y'] = F.di5y.diff() / 100
        X = X.clip(-0.25, 0.25)
    out = []
    for root, r in A.groupby(level=0):
        r = r.droplevel(0).dropna()
        d = r.index.values
        x = X.reindex(r.index).fillna(0).values if F is not None else None
        for t in meses:
            n = d.searchsorted(np.datetime64(t), 'right')
            if n < 147 or (np.datetime64(t) - d[n - 1]) > np.timedelta64(10, 'D'):
                continue
            y = r.values[n - 126:n]
            if x is not None and n >= 378:          # betas estimados no ano ANTES da janela (sem sobreposição)
                A_ = np.column_stack([np.ones(252), x[n - 378:n - 126]])
                b = np.linalg.lstsq(A_, r.values[n - 378:n - 126], rcond=None)[0]
                sd = np.std(r.values[n - 378:n - 126] - A_ @ b)
                idio = y - x[n - 126:n] @ b[1:]     # tira só a parte dos fatores (NÃO o intercepto/drift do ano anterior)
                e = idio - b[0]
            else:                                   # sem fatores (ou pouco histórico): retorno bruto da ação
                idio = e = y
                sd = np.std(r.values[max(0, n - 378):n - 126]) if n >= 200 else np.std(y)
            out.append((root, t, idio.sum() - e[np.abs(e) > 3 * sd].sum()))   # tira os dias de salto (> 3 desvios)
    return pd.DataFrame(out, columns=['root', 't', 'trend6'])


# ------------------------------------------------------------------ variáveis do emissor (para quem não tem ação)
def emissor(D, P):
    P = P.sort_values('t')
    R = D.get('ratings')
    if R is not None:
        R = R.dropna(subset=['data', 'rating_num']).sort_values(['cnpj_raiz', 'data'])
        R['rebaixou'] = R.groupby('cnpj_raiz').rating_num.diff() > 0
        R['ult_rebaix'] = R.data.where(R.rebaixou).groupby(R.cnpj_raiz).ffill()
        cols = ['rating_num'] + [c for c in ('outlook', 'watch') if c in R] + ['ult_rebaix']
        P = pd.merge_asof(P, R.rename(columns={'cnpj_raiz': 'root', 'data': 'k_r'})[['root', 'k_r'] + cols].sort_values('k_r'),
                          left_on='t', right_on='k_r', by='root', direction='backward')
        P['rebaixou_6m'] = ((P.t - P.ult_rebaix).dt.days < 183).astype(float)
    B = D.get('balancos')
    if B is not None:
        B = B.dropna(subset=['data_entrega', 'ativo_total']).query('ativo_total > 0')
        B['alav'] = (B.divida.fillna(0) - B.caixa.fillna(0)) / B.ativo_total
        P = pd.merge_asof(P, B.rename(columns={'cnpj_raiz': 'root', 'data_entrega': 'k_b'})[['root', 'k_b', 'alav']].sort_values('k_b'),
                          left_on='t', right_on='k_b', by='root', direction='backward', tolerance=pd.Timedelta(days=200))
    G = D.get('assembleias')
    if G is not None:
        G = G.dropna(subset=['data_entrega']).sort_values('data_entrega')
        dd = {r_: g.data_entrega.values for r_, g in G.groupby('cnpj_raiz')}
        P['assembleias_180d'] = [dd[r_].searchsorted(np.datetime64(t), 'right')
                                 - dd[r_].searchsorted(np.datetime64(t - pd.Timedelta(days=180)), 'right') if r_ in dd else 0
                                 for r_, t in zip(P.root, P.t)]
    return P


# ------------------------------------------------------------------ regressão logística (numpy, com penalidade igual à usada na pesquisa)
def ajusta(X, y, lam=2.0, it=50):
    X = np.column_stack([np.ones(len(X)), X]); w = np.zeros(X.shape[1]); pen = np.r_[0, np.full(X.shape[1] - 1, lam)]
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ w))
        H = X.T @ (X * (p * (1 - p))[:, None]) + np.diag(pen)
        step = np.linalg.solve(H, X.T @ (p - y) + pen * w)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def prob(w, X):
    return 1 / (1 + np.exp(-np.column_stack([np.ones(len(X)), X]) @ w))


def auc(y, s):
    r = pd.Series(s).rank().values; y = np.asarray(y); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def variaveis(P):
    """Ranks dentro do mês (o modelo compara cada debênture com as outras daquele mês)."""
    base = ['lag3', 'disc'] + [c for c in ('trend6', 'rating_num', 'outlook', 'watch', 'alav', 'assembleias_180d') if c in P]
    for c in base:
        P['r_' + c] = P.groupby('m')[c].rank(pct=True).fillna(0.5) - 0.5
    P['r_disc2'] = P.r_disc ** 2
    for c in ('rating_num', 'alav'):
        if c in P:
            P['tem_' + c] = P[c].notna().astype(float)
    com = ['r_trend6', 'r_lag3', 'r_disc', 'r_disc2']
    sem = ['r_lag3', 'r_disc', 'r_disc2'] + [c for c in ('r_rating_num', 'r_outlook', 'r_watch', 'rebaixou_6m', 'tem_rating_num',
                                                         'r_alav', 'tem_alav', 'r_assembleias_180d') if c in P]
    P['com_acao'] = P.trend6.notna() if 'trend6' in P else False
    return P, {True: com, False: sem}


# ------------------------------------------------------------------ gatilho de curto prazo (negócios e marcas diárias)
def gatilho(D, P, t, diario):
    g = pd.Series(False, index=P.index); motivo = pd.Series('', index=P.index)
    marcas = D['precos_debentures'].dropna(subset=['data', 'pct_par']).sort_values('data')
    N = D.get('negocios')
    if N is not None:
        N = N[(N.data <= t) & (N.data > t - pd.Timedelta(days=30)) & (N.volume_brl >= GATILHO_VOLUME)].sort_values('data')
        ref = marcas.rename(columns={'pct_par': 'marca', 'data': 'dm'})[['codigo', 'dm', 'marca']]
        N = pd.merge_asof(N, ref.assign(dm=ref.dm + pd.Timedelta(days=1)), left_on='data', right_on='dm', by='codigo', direction='backward')
        ruins = set(N.codigo[N.pct_par <= N.marca - GATILHO_NEGOCIO_PTS])
        hit = P.codigo.isin(ruins); g |= hit; motivo[hit] += 'negocio grande abaixo da marca; '
    if diario:                                  # marca de hoje vs ~20 dias úteis atrás (28 dias corridos)
        agora = marcas[(marcas.data <= t) & (marcas.data > t - pd.Timedelta(days=10))].groupby('codigo').pct_par.last()
        antes = marcas[marcas.data <= t - pd.Timedelta(days=28)].groupby('codigo').pct_par.last()
        q = (agora / antes.reindex(agora.index) - 1).dropna()
        caiu = set(q.index[q <= GATILHO_MARCA_20D]); hit = P.codigo.isin(caiu); g |= hit; motivo[hit] += 'marca caiu em 20 dias; '
    return g, motivo.str.rstrip('; ')


# ------------------------------------------------------------------ termômetro de mercado (o "quando")
def termometro(D, P):
    """Fase do mercado. Validado em 2020-26: a taxa de quedas fortes que acabou de acontecer é o melhor termômetro
    (as quedas vêm em sequência; correlação +0,49 com a taxa dos 3 meses seguintes). Amplitude (muitas DI caindo) e
    emissão primária fraca ajudam pouco além disso (reforço). Contar alarmes (dólar, juros, Ibov...) NÃO funcionou."""
    ult = P.m.max()
    taxa = P.groupby('m').queda.mean().dropna()
    recente, normal = taxa.tail(3).mean(), taxa.iloc[:-3].mean() if len(taxa) > 9 else taxa.mean()
    px = P.set_index(['codigo', 'm']).pct_par
    di = P[P.ix == 'DI']
    queda_mes = (di.pct_par.values - px.reindex(list(zip(di.codigo, di.m - 1))).values) <= -0.25
    amp = pd.Series(queda_mes, index=di.index).groupby(di.m).mean()
    a_ult = amp.get(ult, np.nan)
    a_alarme = a_ult >= amp[amp.index < ult].quantile(0.8) if (amp.index < ult).sum() >= 12 else False
    linhas = [f'Termômetro do mercado ({ult}):',
              f'  quedas fortes recentes: {recente:.1%} das debêntures (meses {taxa.index[-3]} a {taxa.index[-1]}) vs normal {normal:.1%}'
              f'  -> {"FASE RUIM: agir sobre a lista vermelha" if recente > 1.25 * normal else "fase normal"}',
              f'  amplitude (DI caindo >= 0,25 pt no mês): {a_ult:.0%} -> {"alarme (acima do percentil 80 histórico)" if a_alarme else "sem alarme"}']
    cad = D['cadastro_debentures']
    if {'data_emissao', 'volume_emissao'} <= set(cad):
        iss = cad.dropna(subset=['data_emissao']).groupby(cad.data_emissao.dt.to_period('M')).volume_emissao.sum()
        iss = iss.reindex(pd.period_range(iss.index.min(), ult, freq='M'), fill_value=0)
        r = (iss.rolling(3).sum() / 3 / iss.rolling(12).mean()).dropna()
        if len(r) > 12:
            fraca = r.iloc[-1] <= r.iloc[:-1].quantile(0.2)
            linhas.append(f'  emissão primária (3m vs média 12m): {r.iloc[-1]:.2f} -> {"alarme (fraca)" if fraca else "sem alarme"}')
    print('\n'.join(linhas))
    return linhas


def avalia_gatilho(D, P, feats, diario):
    """Backtest do gatilho: dentro do top 10% fora da amostra, taxa de queda quando o gatilho disparou vs não."""
    if not diario and D.get('negocios') is None:
        print('\nGatilho: sem dados diários (marcas diárias ou negócios) - não avaliado.')
        return
    P = P.copy(); P['p'] = np.nan
    for lst, cols in feats.items():
        for y in range(int(P.ano.min()) + 2, int(P.ano.max()) + 1):
            tr = P[(P.com_acao == lst) & (P.m <= pd.Period(f'{y}-01', 'M') - 4) & P.queda.notna()]
            te = (P.com_acao == lst) & (P.ano == y)
            if tr.queda.sum() >= 20 and te.any():
                P.loc[te, 'p'] = prob(ajusta(tr[cols].values, tr.queda.values), P.loc[te, cols].values)
    P['pct'] = P.groupby('m').p.rank(pct=True)
    T = P[(P.pct >= 0.9) & P.queda.notna()].copy()
    T['gat'] = False
    for m, idx in T.groupby('m').groups.items():
        T.loc[idx, 'gat'] = gatilho(D, T.loc[idx], m.to_timestamp(how='end').normalize(), diario)[0].values
    print('\nGatilho dentro do top 10% (fora da amostra), por ano: taxa de queda com gatilho / sem gatilho (n com gatilho)')
    for y, g in T.groupby('ano'):
        a, b = g[g.gat], g[~g.gat]
        print(f'    {y}: {a.queda.mean():.1%} / {b.queda.mean():.1%}  (n {len(a)})' if len(a) else f'    {y}: gatilho não disparou')
    a, b = T[T.gat], T[~T.gat]
    print(f'  total: {a.queda.mean():.1%} com gatilho ({len(a)}) vs {b.queda.mean():.1%} sem ({len(b)})')


# ------------------------------------------------------------------ principal
def prepara():
    print('Lendo dados_modelo/ ...')
    D = {k: le(k) for k in ARQUIVOS}
    P, diario = painel(D)
    T = tendencia(D, sorted(P.t.unique()))
    P = P.merge(T, on=['root', 't'], how='left')
    P = emissor(D, P)
    P = P[P.lag3.notna() & P.disc.notna()].copy()
    P['ano'] = P.m.dt.year
    P, feats = variaveis(P)
    print(f'Painel: {len(P):,} debênture-meses, {P.codigo.nunique()} debêntures, {P.m.min()} a {P.m.max()}, '
          f'{P.com_acao.mean():.0%} com ação, quedas fortes {P.queda.mean():.1%}')
    return D, P, feats, diario


def backtest(P, feats):
    print('\nQualidade fora da amostra (AUC: 0,5 = chute, 1 = perfeito; taxa de queda no top 10% vs geral):')
    for lst, cols in feats.items():
        S = P[P.com_acao == lst]
        print(f'  {"COM ação" if lst else "SEM ação"} (variáveis: {", ".join(c.replace("r_", "") for c in cols)})')
        for y in range(int(S.ano.min()) + 2, int(S.ano.max()) + 1):
            tr = S[(S.m <= pd.Period(f'{y}-01', 'M') - 4) & S.queda.notna()]; te = S[(S.ano == y) & S.queda.notna()]
            if te.queda.sum() < 5 or tr.queda.sum() < 20:
                continue
            p = prob(ajusta(tr[cols].values, tr.queda.values), te[cols].values)
            top = p >= np.quantile(p, 0.9)
            print(f'    {y}: AUC {auc(te.queda.values, p):.2f} | queda no top 10% {te.queda[top].mean():.1%} vs geral {te.queda.mean():.1%}'
                  f' | {int(te.queda.sum())} quedas')


def lista(D, P, feats, diario, termo=()):
    ult = P.m.max(); t = P.t[P.m == ult].iloc[0]
    N = P[P.m == ult].copy(); N['prob'] = np.nan
    for lst, cols in feats.items():
        tr = P[(P.com_acao == lst) & (P.m <= ult - 3) & P.queda.notna()]
        sel = N.com_acao == lst
        if sel.any() and tr.queda.sum() >= 20:
            N.loc[sel, 'prob'] = prob(ajusta(tr[cols].values, tr.queda.values), N.loc[sel, cols].values)
    N['percentil'] = N.prob.rank(pct=True)
    N['faixa'] = np.select([N.percentil >= 0.9, N.percentil >= 0.7], ['VERMELHO', 'AMARELO'], 'normal')
    N['gatilho'], N['motivo_gatilho'] = gatilho(D, N, t, diario)
    if D.get('posicoes') is not None:
        pos = D['posicoes'].pivot_table(index='codigo', columns='fundo', values='valor_brl', aggfunc='sum')
        N = N.join(pos, on='codigo'); N['total_fundos_brl'] = pos.sum(axis=1).reindex(N.codigo).values
    if 'trend6' in N:
        N['acao_6m_%'] = (np.exp(N.trend6) - 1) * 100
    N['lag3_%'], N['desconto_pts'] = N.lag3 * 100, N.disc
    cols = [c for c in ['emissor', 'root', 'codigo', 'ix', 'pct_par', 'faixa', 'prob', 'percentil', 'gatilho', 'motivo_gatilho',
                        'com_acao', 'acao_6m_%', 'lag3_%', 'desconto_pts', 'rating_num', 'outlook', 'rebaixou_6m', 'alav',
                        'assembleias_180d', 'total_fundos_brl'] if c in N] + \
        ([c for c in D['posicoes'].fundo.unique() if c in N] if D.get('posicoes') is not None else [])
    N = N.sort_values('prob', ascending=False)[cols]
    SAIDA.mkdir(exist_ok=True)
    f = SAIDA / f'lista_risco_{ult}.csv'
    N.round(4).to_csv(f, sep=';', decimal=',', index=False, encoding='utf-8-sig')
    (SAIDA / f'termometro_{ult}.txt').write_text('\n'.join(termo), encoding='utf-8')
    print(f'\nLista de {ult} salva em {f}  ({(N.faixa == "VERMELHO").sum()} vermelhas, {(N.faixa == "AMARELO").sum()} amarelas, '
          f'{int(N.gatilho.sum())} com gatilho)')
    print(N.head(25).to_string(index=False, float_format=lambda v: f'{v:,.2f}'))


def modelos():
    DADOS.mkdir(exist_ok=True)
    txt = ['ARQUIVOS DE ENTRADA (um CSV por item, nesta pasta). Obrigatórios: precos_debentures, cadastro_debentures.\n']
    for k, (ob, cols, opc, desc) in ARQUIVOS.items():
        f = DADOS / f'{k}.csv'
        if not f.exists():
            f.write_text(';'.join(cols + opc) + '\n', encoding='utf-8')
        txt.append(f'{k}.csv  ({"OBRIGATÓRIO" if ob else "opcional"})\n  colunas: {"; ".join(cols)}'
                   + (f'\n  opcionais: {"; ".join(opc)}' if opc else '') + f'\n  {desc}\n')
    (DADOS / 'LEIA-ME.txt').write_text('\n'.join(txt), encoding='utf-8')
    print(f'Modelos criados em {DADOS}. Leia LEIA-ME.txt.')


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('--modelos', action='store_true'); a.add_argument('--backtest', action='store_true')
    a = a.parse_args()
    if a.modelos:
        modelos()
    else:
        D, P, feats, diario = prepara()
        if a.backtest:
            backtest(P, feats)
            avalia_gatilho(D, P, feats, diario)
        termo = termometro(D, P)
        lista(D, P, feats, diario, termo)
