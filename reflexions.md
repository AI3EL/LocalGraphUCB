Parfois on a de bons resultats pour le degree regret et parfois pour le real regret. 
Exemple:
SBM1.png: SimpleSBM(0.001, 0.1, [20,20,40,20]) -> bon degree_regret, mauvais cc_regret

Quand on obtient quasiment tout le temps une composante connexe, le score est quasiment toujours maximal peu 
importe le choix du noeud. 
Voir images de Louis: 0.01, 0.1, 50,50,100,50 udb double
SBM2: graph = SimpleSBM(0.05, 0.2, [10,10,20,10])
SBM3graph = SimpleSBM(0.1, 0.7, [10,10,20,10])

SBM12 vs SBM15 --> parfois il faut juste rajouter du temps pour que degree_regret devienne bon

En regime supercritique (quasiment tout le temps 1 cc), prendre n'importe quel noeud donne une reward maximale

La borne sup necessite de connaitre les c_a, donc possible en les estimant.

## Contre-exemple sans l'assumption 1 et avec l'assumption 2:

L'idee intuitive est que pour minimiser le degré tout en maximisant la composante connexe, il faut faire une chaine.

Conditions:
n blocs:
Ki,i = 1
Ki,i+1 = Ki+1,i = 1 pour i <= n-2
le reste vaut 0
(Ca marche aussi pour 'presque' 0 et 'presque' 1)
En gros on a une chaine pour les n-1 premiers blocs et le dernier bloc est isolé.

Et:
N_i < N_n /3 pour tout i < n
sum(N_j, j<n) > N_n
Du coup pour i < n: mu_i = N_i-1 + N_i + N_i+1 et mu_n = N_n
Mais c_i = sum(N_j, j<n) et c_n = N_n

## Idee de variation: connaitre les voisins (au lieu de juste connaitre leur nombre)

Dans le SBM par exemple ca permettrait de contrer le contre-exemple: on doit pouvoir récupérer empiriquement les parametre Ki,j et savoir a peu pres qui appartient a quoi ... 
PB: meme quand on connait les Ki,j peut on savoir quel point il faut prendre pour maximiser la reward ? La solution se situe peut-être dans les appendices du papier. Solution naive: simuler les c_i avec des simulations de graphes avec les K_i,j^hat
Q: Chung-Lu ? Modelfree ?

## Idee de variation 2: connaitre le nombre de k-voisins (voisins à distance k)

Idee: utiliser les k-voisins permet de contrer l'exemple de la chaine dans certains cas. 
Sans doute difficile a formaliser... 
