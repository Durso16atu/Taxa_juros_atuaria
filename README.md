# Macro LLM - Modelo de Linguagem Causal (BCB & Atuária)

Modelo de linguagem causal (Decoder-Only Transformer no estilo GPT / nanoGPT) com ~12,3M de parâmetros, treinado do zero sobre um corpus especializado de finanças, política monetária e regulação prudencial brasileira.

## Especificações Técnicas
- **Arquitetura**: Transformer Causal (6 blocos, 6 cabeças de atenção, dimensão $d_{\text{model}} = 384$, contexto de 256 tokens, dropout 0.2).
- **Otimização**: AdamW com Cosine Decay, warmup linear e FlashAttention (`F.scaled_dot_product_attention`).
- **Tokenização**: ByteLevel BPE ($V = 4.096$, taxa de compressão de 3,89 caracteres/token).
- **Base de Treinamento**: 14,1 milhões de caracteres higienizados (~3,6M tokens) particionados de forma balanceada a partir de 13.655 blocos documentais cobrindo:
  1. Atas e Comunicados do Copom (Banco Central do Brasil)
  2. Relatórios de Política Monetária (RPM / RI - 19 edições completas)
  3. Relatórios de Estabilidade Financeira (REF - 6 edições completas)
  4. Resoluções e Circulares de Solvência/Provisões da SUSEP

## Resultados de Treinamento
- **Loss Inicial**: 8,35 (consistente com a entropia teórica $\ln(4096) \approx 8,32$)
- **Loss de Treino Final**: 2,1511
- **Loss de Validação Final**: 2,2903
- **Diferença de Generalização**: 0,1392 (sem overfitting significativo)
