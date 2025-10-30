```mermaid
%%{init: {"theme":"base","themeVariables":{"fontSize":"22px"}}}%%
flowchart LR
  U["Input"] --> TKN["Tokenizer"]
  TKN --> EMB["Embedding"]
  EMB --> GATE["Neuroplastic Gating"]

  subgraph MECH["Plasticity Signals"]
    direction TB
    HEBB["Hebbian"]
    HOMEO["Homeostasis"]
    STDP["STDP"]
    CONS["Consolidation"]
    REW["Reward Modulation"]
  end
  MECH -- "modulate" --> GATE

  subgraph EXP["Expert Pool"]
    direction LR
    E1["Expert A"]
    E2["Expert B"]
    E3["Expert C"]
    GROW["Dynamic Growth / Pruning"]
  end

  GATE -- "top-k routing" --> E1
  GATE --> E2
  GATE --> E3

  E1 --> MIX["Weighted Mixture"]
  E2 --> MIX
  E3 --> MIX
  MIX --> CTX["Context Vector"]
  CTX --> DEC["Decoder"]
  DEC --> OUT["Output"]

  CHK["Expert Checkpoints"] --- EXP
  GROW -. "adds/prunes" .-> EXP

  ```