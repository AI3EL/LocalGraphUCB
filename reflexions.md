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

