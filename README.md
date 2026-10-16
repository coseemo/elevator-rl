# Elaborato — Autonomous Agents and Intelligent Robotics

## Descrizione del progetto

Questo repository contiene il progetto sviluppato per il corso di **Autonomous Agents and Intelligent Robotics**.

L'obiettivo dell'elaborato è simulare e analizzare l'applicazione di algoritmi di **Reinforcement Learning (RL)** per automatizzare la gestione di un ascensore all'interno di un edificio di cinque piani.

Il lavoro è organizzato in tre parti principali:

1. **Definizione dell'ambiente:** modellazione del comportamento dell'ascensore, dei passeggeri e delle code di attesa.
2. **Fondamenti teorici:** introduzione all'algoritmo di Reinforcement Learning utilizzato.
3. **Addestramento e analisi:** studio del processo di apprendimento e ottimizzazione degli iperparametri.

Il progetto si conclude con un'analisi dei principali risultati e dei fenomeni osservati durante lo sviluppo e la sperimentazione del modello.

## Struttura del repository

```text
ascensori-rl/
├── Elevator.ipynb
├── .gitignore
└── README.md
```

Il notebook `Elevator.ipynb` contiene la definizione dell'ambiente, i test di verifica e le successive fasi di analisi e addestramento; le dipendenze sono direttamente installabili dal notebook.

## Risorse

- [Documentazione ufficiale di Gymnasium](https://gymnasium.farama.org/)

---

# 1. Modellazione dell'ambiente — Alti e Bassi

## 1.1 Descrizione del sistema

L'ambiente simulato rappresenta un edificio di cinque piani, servito da un ascensore in grado di trasportare un solo passeggero alla volta.

Per ciascun piano viene mantenuta una coda di passeggeri in attesa. Gli arrivi sono generati casualmente mediante una distribuzione di Poisson, che determina il numero di nuovi passeggeri che raggiungono ciascun piano durante ogni intervallo temporale.

La destinazione di ciascun passeggero viene estratta al momento dell'imbarco, utilizzando una matrice di probabilità origine-destinazione \(P_{OD}\).

Ogni step della simulazione corrisponde a un intervallo temporale di **5 secondi**. Durante ciascuno step l'agente seleziona un'azione, l'ascensore aggiorna la propria posizione e vengono gestiti gli eventuali imbarchi, sbarchi e nuovi arrivi.

## 1.2 Spazio delle osservazioni

Lo stato osservabile dell'ambiente è definito come:

$$
S_t =
\{
q_0,\ldots,q_4,\,
w_0,\ldots,w_4,\,
\ell_t,\,
d_t,\,
p_t
\}
$$

Dove:

- \(q_i\) rappresenta il numero di passeggeri in attesa al piano \(i\).
- \(w_i\) rappresenta il tempo di attesa, in secondi, del passeggero che aspetta da più tempo al piano \(i\). Se la coda è vuota, il valore è pari a zero.
- \(\ell_t\) rappresenta il piano attuale dell'ascensore.
- \(d_t\) rappresenta il piano di destinazione del passeggero a bordo.
- \(p_t \in \{0,1\}\) indica se l'ascensore è occupato.

L'osservazione permette quindi di conoscere sia la numerosità delle code sia l'anzianità della maggiore attesa presente a ciascun piano. Queste informazioni sono complementari: il solo numero di passeggeri non permette di distinguere una coda appena formata da una in cui qualcuno aspetta da molto tempo.

Quando l'ascensore è vuoto, il piano di destinazione coincide con quello attuale. Nel notebook, le osservazioni relative alle code e ai tempi di attesa sono rappresentate rispettivamente dai campi `q1`–`q5` e `w1`–`w5`.

## 1.3 Spazio delle azioni e dinamica dell'ascensore

Le azioni disponibili all'agente sono:

$$
A = \{\text{Resta fermo},\ \text{Sale},\ \text{Scende}\}
$$

Per ogni azione di salita o discesa, l'ascensore si sposta di un piano, rispettando i limiti dell'edificio.

L'imbarco avviene quando l'ascensore è vuoto, si trova a un piano con almeno un passeggero in attesa e si verifica una delle seguenti condizioni:

- l'agente sceglie di restare fermo;
- un passeggero è appena sbarcato al piano corrente.

Lo sbarco avviene automaticamente quando l'ascensore raggiunge il piano di destinazione.

Questa dinamica consente all'ascensore di attraversare un piano senza necessariamente fermarsi per imbarcare un passeggero. Inoltre, quando un passeggero scende in un piano in cui sono presenti persone in attesa, il ricambio dei passeggeri avviene nello stesso step.

## 1.4 Definizione della reward

L'obiettivo è rendere efficiente la gestione delle code, limitando sia il numero di passeggeri in attesa sia il tempo trascorso in coda.

Una prima possibilità consiste nel penalizzare il numero totale di passeggeri in attesa:

$$
R_t^{(1)} = -N_t
$$

dove \(N_t\) rappresenta il numero complessivo di passeggeri in coda. Questa formulazione penalizza ogni passeggero in attesa in modo uniforme.

Una seconda possibilità consiste nel penalizzare la somma dei tempi di attesa:

$$
R_t^{(2)} =
-\sum_{i \in Q_t}(t-t_i^{\text{arrivo}})
$$

dove \(Q_t\) è l'insieme dei passeggeri ancora in attesa, \(t\) è il tempo corrente della simulazione e \(t_i^{\text{arrivo}}\) è l'istante di arrivo del passeggero \(i\).

In questo caso, i passeggeri che aspettano da più tempo contribuiscono maggiormente alla penalità. Tuttavia, utilizzare esclusivamente il tempo di attesa accumulato può rendere la reward fortemente influenzata dalle attese più lunghe.

Per combinare entrambe le esigenze, è stata definita una reward che considera sia il numero di passeggeri in attesa sia la somma dei loro tempi di attesa:

$$
R_t =
-\left(
N_t + \lambda \frac{W_t}{T_{\mathrm{rif}}}
\right)
$$

Dove:

- \(N_t\) è il numero totale di passeggeri ancora in coda.
- \(W_t\) è la somma dei loro tempi di attesa, espressa in secondi.
- \(T_{\mathrm{rif}}\) è un tempo di riferimento espresso in secondi.
- \(\lambda\) è un parametro che regola il peso attribuito alle attese prolungate.

La normalizzazione tramite \(T_{\mathrm{rif}}\) permette di rendere confrontabili le due componenti della reward. In questo modo viene penalizzata sia la presenza di code numerose sia la permanenza prolungata dei passeggeri in attesa.

La reward viene calcolata sullo stato risultante dalla transizione, dopo la gestione degli imbarchi e degli eventuali nuovi arrivi.

## 1.5 Definizione dell'ambiente con Gymnasium

Per implementare la simulazione è stata utilizzata la libreria Python **Gymnasium**, che fornisce un'interfaccia standard per la definizione di ambienti di Reinforcement Learning.

L'ambiente è stato implementato mediante la classe `ElevatorEnv`, che eredita da `gym.Env`. Questa struttura permette di separare la logica dell'ambiente da quella dell'agente e di utilizzare strumenti compatibili con l'interfaccia di Gymnasium.

All'interno della classe sono stati implementati i seguenti metodi principali:

- `__init__()`: inizializza l'ambiente, definendo le caratteristiche dell'edificio, le code dei passeggeri, il processo di arrivo e gli spazi delle azioni e delle osservazioni.
- `reset()`: inizializza un nuovo episodio, ripristinando lo stato dell'ascensore, le code e le informazioni accumulate durante la simulazione precedente.
- `step(action)`: esegue un passo della simulazione in base all'azione selezionata dall'agente. Gestisce il movimento dell'ascensore, gli imbarchi, gli sbarchi, gli arrivi e il calcolo della reward.

Il metodo `step()` restituisce cinque elementi, secondo l'interfaccia di Gymnasium: `observation`, `reward`, `terminated`, `truncated` e `info`.

In particolare, `terminated` indica il raggiungimento di una condizione terminale naturale, mentre `truncated` segnala l'interruzione dell'episodio dovuta al raggiungimento del numero massimo di step. Nel modello viene utilizzato il troncamento, poiché si considera un flusso di passeggeri continuo e si limita la durata di ciascun episodio.

## 1.6 Testing dell'ambiente

Per verificare la correttezza dell'ambiente sono stati eseguiti i seguenti test:

1. Verifica della compatibilità con Gymnasium tramite `check_env()`.
2. Verifica del comportamento delle azioni.
3. Verifica del processo di arrivo dei passeggeri.
4. Verifica dell'imbarco.
5. Verifica dello sbarco.
6. Verifica dello sbarco seguito dall'imbarco automatico nello stesso step.
7. Verifica del calcolo della reward.
8. Verifica della validità delle osservazioni.
9. Verifica della coerenza dello stato dell'ambiente.

I test sono stati utilizzati per controllare il funzionamento delle principali transizioni e la coerenza tra le code, i tempi di arrivo e lo stato dell'ascensore.
