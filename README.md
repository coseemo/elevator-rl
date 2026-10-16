# Elaborato Autonomous Multi Agent

Questo notebook contiene il progetto per il corso di Autonomous Agents and Intelligent Robotics.

Per questo elaborato si è simulare e analizzato l'applicazione dell'algoritmo di Reinforcement Learning **placeholder** per andare ad automatizzare in modo "intelligente" il comportamento di un ascensore; il lavoro è diviso in tre parti principali:

1. La definizione del comportamento dell'ascensore e del relativo ambiente virtuale.
2. Un richiamo teorico su **placeholder**.
3. Un'analisi approfondita sull'addestramento e il tuning degli iperparametri del modello.

Infine, è inoltre presente una piccola sezione di conclusioni che riassume i fenomeni salienti osservati durante lo sviluppo del progetto.

## Albero Cartelle

```text
ascensori-rl/
├── Elevator.ipynb
├── .gitignore
└── README.md
```

## Risorse Utilizzate

- https://gymnasium.farama.org/

---

## Prima Parte: Alti e Bassi

In questa sezione del notebook andiamo a definire il comportamento dell'ascensore e dell'ambiente che questo andrà a servire: un palazzo di 5 piani e i suoi, più o meno numerosi, passeggeri.

Per definire tale ambiente è stata disposta una coda per ciascun piano del palazzo che modella la coda di passeggeri in attesa davanti all'ascensore, i passeggeri vengono generati in modo casuale tramite una distribuzione di Poisson, la quale, durante ogni intervallo temporale, determina il numero di nuovi arrivi a ciascun piano. La destinazione di ciascun passeggero viene invece estratta quando quest'ultimo sale sull'ascensore, utilizzando una matrice di probabilità origine-destinazione `P_{OD}`.

Lo stato osservabile dell'ambiente sarà dunque composto come segue:

$$
S = \{c0, c1, c2, c3, c4,
w0, w1, w2, w3, w4,
piano\ attuale, piano\ destinazione, passeggero\ a\ bordo\}
$$


Per ogni piano l'agente osserva dunque sia **quanti passeggeri** sono in coda(C) sia **da quanto tempo aspetta il passeggero che è arrivato per primo**(W): `attesa_i` è infatti il tempo di attesa, in secondi, del passeggero che aspetta da più tempo al piano `i` (se la coda è vuota, è pari a 0). Il campo relativo alla destinazione rappresenta la destinazione del passeggero a bordo, quando l'ascensore è vuoto questo coincide con il piano attuale dell'ascensore. L'ascensore può inoltre trasportare **un solo passeggero** alla volta, per questo il campo relativo al passeggero a bordo può assumere solo i valori 0 e 1.

L'ascensore si sposta di un piano per ogni azione di salita o discesa e ciascuno step della simulazione corrisponde a un intervallo temporale di 5 secondi. L'imbarco avviene quando l'ascensore è vuoto **e** si trova a un piano con almeno un passeggero in attesa **e** |l'agente sceglie di restare fermo **o** ha appena sbarcato un altro passeggero|, lo sbarco, invece, avviene automaticamente quando l'ascensore raggiunge il piano di destinazione; in questo modo si è cercato di modellare sia la possibilità di transitare da un piano con passeggeri in attesa **senza necessariamente fermarsi** sia il fatto che, una volta portato a destinazione un passeggero in un piano con altri passeggeri in attesa, lo scambio tra chi esce e chi entra avvenisse in un **solo step**.

Ciò che vogliamo rendere efficiente è la gestione delle code ai vari piani, in particolare vorremo che le code non fossero troppo lunghe ma neanche che vi fossero persone che aspettano da troppo tempo; se si usasse come reward `r =-N` avremmo code meno numerose ma, potrebbe capitare di avere passeggeri che aspettano molto tempo in attesa, se, al contrario, si usasse `R_t = -\sum_{i \in Q_t}(t-t_i^{arrivo})` (dove `Q_t` rappresenta l'insieme dei passeggeri ancora in attesa al termine dello step), si penalizzerebbero maggiormente le attese lunghe, questo però potrebbe portare a trascurare maggiormente le code più lunghe, le quali sul lungo periodo potrebbero causare un rallentamento.

Per questo motivo, si è cercato di definire una reward che considerasse entrambe queste esigenze:

$$
R_t = -\left(N_t + \lambda \frac{W_t}{T_{rif}}\right)
$$

dove `W_t` rappresenta la somma dei tempi di attesa di tutti i passeggeri ancora in coda, `T_{rif}` è un tempo di riferimento espresso in secondi e `\lambda` è un parametro che regola il peso attribuito alle attese prolungate.

In questo modo, la reward penalizza sia la presenza di numerosi passeggeri in attesa sia il protrarsi delle attese nel tempo. Il parametro `\lambda` permette di regolare il contributo del secondo termine, mentre la normalizzazione tramite `T_{rif}` rende le due componenti più facilmente confrontabili.

Infine, le possibili azioni dell'ascensore sono le seguenti:

$$
A = \{Resta\ fermo,\ Sale,\ Scende\}
$$

### Gymansium: Costruendo la Palestra

Per modellare l'ambiente è stata utilizzata la libreria Python **Gymnasium**, che mette a disposizione un'interfaccia standard per la definizione di ambienti per il *Reinforcement Learning* (RL), in questo modo è possibile separare la logica dell'ambiente da quella dell'agente e utilizzare algoritmi di apprendimento compatibili con la libreria.

Per costruire l'ambiente è stata definita la classe `ElevatorEnv`, ereditando `gym.Env`, questo ci consente di avere accesso alle funzionalità di base previste da Gymnasium e viene garantita la compatibilità con gli strumenti che implementano algoritmi di RL.

All'interno della classe sono stati implementati i seguenti metodi principali:

- `__init__()`: viene inizializzato l'ambiente, definendo le caratteristiche dell'edificio, le code dei passeggeri, i parametri del processo di arrivo e gli spazi delle azioni e delle osservazioni.
- `reset()`: viene inizializzata una nuova simulazione, ripristinando le condizioni iniziali dell'ambiente e azzerando le informazioni accumulate durante l'episodio precedente. Vengono inoltre restituite la prima osservazione e un dizionario contenente informazioni aggiuntive.
- `step(action)`: viene eseguito un passo della simulazione sulla base dell'azione selezionata dall'agente. In questo metodo vengono gestiti lo spostamento dell'ascensore, l'imbarco e lo sbarco dei passeggeri, l'arrivo di nuovi passeggeri e il calcolo della reward.

Al termine di ogni chiamata al metodo `step()`, vengono restituiti cinque elementi, secondo l'interfaccia prevista da Gymnasium: `observation`, `reward`, `terminated`, `truncated` e `info`. In particolare, `terminated` indica il raggiungimento di una condizione terminale dell'ambiente, mentre `truncated` segnala l'interruzione dell'episodio dovuta al raggiungimento del numero massimo di step previsto; in questo caso verrà utilizzato `truncated` in quanto si è immaginato che il flusso dei passeggeri fosse continuo.

Una volta definiti i metodi principali e gli spazi delle azioni e delle osservazioni, l'ambiente può essere istanziato e utilizzato per l'addestramento di un agente. Durante l'apprendimento, l'agente riceve le osservazioni dell'ambiente, seleziona le azioni da eseguire e ottiene una reward in funzione delle conseguenze delle proprie decisioni.

#### Testing dell'ambiente

Vengono adesso svolti i seguenti test sull'ambiente definito sinora al fine di verificarne la correttezza:

1. test di correttezza sintattica con *check_env()* da gymnasium
2. verifica del funzionamento delle azioni
3. verifica dell'arrivo dei passeggeri
4. verifica dell'imbarco
5. verifica dello sbarco
6. verifica sbarco-imbarco
7. verifica della reward
8. verifica dell'osservazione
9. test di coerenza dello stato

L'ambiente supera tutti i test sopra elencati.
