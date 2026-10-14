---
title: "Rechnen mit schwingenden Ionen: Wie ein Quantencomputer in der Penning-Falle funktioniert"
description: "Einzelne Beryllium-Ionen schweben in einem starken Magnetfeld über einem Chip, zittern auf Nanometern hin und her – und rechnen dabei. Dieser Beitrag erklärt ohne Formeln, wie das Zürcher ETH-Spin-off ZuriQ Quantencomputer aus gefangenen Ionen baut, warum ausgerechnet die Schwingungen der Ionen – die sogenannten Phononen – der Schlüssel zum Rechnen sind, und weshalb ein Quantenalgorithmus namens Shor-Algorithmus die heutige Verschlüsselung herausfordert."
pubDate: "October 10 2026"
heroImage: "/personal_blog/PenningTrapLogo.webp"
---
 
*Author: Prof. Dr. Christoph Würsch, Institute for Computational Engineering ICE, Eastern Switzerland University of Applied Sciences OST*

[Download:](/personal_blog/quantencomputer_penning_traps_WUCH.pdf)

# Rechnen mit schwingenden Ionen: Wie ein Quantencomputer in der Penning-Falle funktioniert

*Einzelne Beryllium-Ionen schweben in einem starken Magnetfeld über einem Chip, zittern auf Nanometern hin und her – und rechnen dabei. Dieser Beitrag erklärt ohne Formeln, wie das Zürcher ETH-Spin-off ZuriQ Quantencomputer aus gefangenen Ionen baut, warum ausgerechnet die Schwingungen der Ionen – die sogenannten Phononen – der Schlüssel zum Rechnen sind, und weshalb ein Quantenalgorithmus namens Shor-Algorithmus die heutige Verschlüsselung herausfordert.*

---

## Ein Computer aus einzelnen Atomen

Ein gewöhnlicher Computer speichert Information in Bits, die entweder 0 oder 1 sind. Ein Quantencomputer verwendet **Qubits**: winzige physikalische Systeme mit zwei unterscheidbaren Zuständen, die sich – solange niemand hinschaut – auch in einer *Überlagerung* beider Zustände befinden können. Oft liest man, ein Qubit sei „gleichzeitig 0 und 1“. Das ist nur die halbe Wahrheit. Wichtiger ist, dass sich Überlagerungen wie Wellen verhalten: Zwei Rechenwege, die zum selben Ergebnis führen, können sich gegenseitig verstärken – oder auslöschen, wie zwei Wasserwellen, bei denen ein Wellenberg auf ein Wellental trifft. Ein guter Quantenalgorithmus ist so gebaut, dass sich die falschen Antworten auslöschen und die richtigen übrig bleiben.

Für Qubits gibt es viele Kandidaten: supraleitende Schaltkreise, Lichtteilchen, Elektronen in Halbleitern – und einzelne elektrisch geladene Atome, sogenannte **Ionen**. Ionen haben einen unschlagbaren Vorteil: Jedes Beryllium-Ion ist exakt gleich wie jedes andere. Die Natur liefert perfekt identische Qubits. Man muss sie „nur“ festhalten, kühlen und präzise steuern.

![Animation: Drei Ionen in einer linearen Anordnung von Penning-Mikrofallen werden gehalten und transportiert.](personal_blog/RechnenQbitPenningFalle.png)

## Wie hält man ein einzelnes Atom fest?

Ein Ion ist elektrisch geladen, also liegt es nahe, es mit elektrischen Feldern festzuhalten. Doch hier stellt sich die Physik quer: Mit rein statischen elektrischen Feldern lässt sich ein geladenes Teilchen nie in allen drei Richtungen gleichzeitig einsperren. Man kann eine Mulde bauen, die in einer Richtung hält – dann ist das Potential in einer anderen Richtung zwangsläufig ein Hügel. Man spricht von einem Sattel, wie bei einem Reitsattel: Nach vorne und hinten geht es bergauf, zur Seite bergab.

Die meisten heutigen Ionen-Quantencomputer umgehen das mit schnell wechselnden Hochfrequenzfeldern, die den Sattel so rasch drehen, dass das Ion nicht herunterfallen kann. Die **Penning-Falle**, auf die ZuriQ setzt, wählt einen anderen, eleganten Weg: Ein statisches elektrisches Feld hält das Ion in der Höhe fest, und ein sehr starkes Magnetfeld – rund 3 Tesla, etwa 60 000-mal stärker als das Erdmagnetfeld – verhindert, dass es zur Seite wegrollt. Ein bewegtes geladenes Teilchen wird im Magnetfeld nämlich immer seitlich abgelenkt. Statt vom Sattel zu rutschen, läuft das Ion deshalb auf kleinen Kreisbahnen um seinen Platz herum [1], [2].

Die Gruppe von Jonathan Home an der ETH Zürich hat 2024 gezeigt, dass dieses Prinzip auch auf einem **Mikrochip** funktioniert: Ein einzelnes Beryllium-Ion schwebt rund 150 Mikrometer – etwa die Dicke von zwei Haaren – über der Chipoberfläche. Es lässt sich vollständig quantenmechanisch kontrollieren und in beliebige Richtungen über den Chip fahren [1]. Weil keine Hochfrequenzfelder nötig sind, kann man Fallen im Prinzip überall auf dem Chip erzeugen. Das ETH-Spin-off **ZuriQ** baut darauf eine zweidimensionale Architektur auf. Nach eigenen Angaben hat das Unternehmen ein Gitter aus 3×3 einzeln kontrollierten Ionen demonstriert; die Chips werden zusammen mit Infineon gefertigt, das Ziel sind Hunderte von Qubits auf einem Chip [3].

Die folgende Animation zeigt das Prinzip. Unten liegen die Elektroden des Chips, deren Spannungen (rot positiv, blau negativ) die Fallen formen. Die Ionen schweben darüber und schwingen leicht auf und ab. Von oben gesehen laufen sie auf kleinen Schleifenbahnen – das ist die Wirkung des Magnetfelds. Das kleine Diagramm rechts unten zeigt den Sattel: In der Höhe sitzt das Ion in einer Mulde, entlang des Chips auf einem Hügel. Und schliesslich sieht man, wie ZuriQ Ionen bewegt: Ändert man die Elektrodenspannungen langsam, wandert die Falle mit – und das Ion mit ihr. So lassen sich zwei Ionen zusammenführen, die miteinander rechnen sollen.

![Animation: Drei Ionen in einer linearen Anordnung von Penning-Mikrofallen werden gehalten und transportiert.](personal_blog/penning_falle_1d.gif)

*Animation 1: Drei Ionen in Penning-Mikrofallen über einem Chip (vereinfachtes Modell, Bewegungen stark vergrössert und verlangsamt). Oben: Seitenansicht mit Elektroden und Potential. Unten links: Draufsicht mit den Kreisbahnen im Magnetfeld. Unten rechts: der Sattelpunkt am Ort eines Ions.*

## Was genau ist das Qubit?

Ein Beryllium-Atom hat vier Elektronen; dem Ion fehlt eines. Zwei der drei verbleibenden Elektronen sind fest gebunden und spielen keine Rolle. Das dritte, das äusserste Elektron, besitzt einen **Spin** – man kann ihn sich vereinfacht als winzige Kompassnadel vorstellen. Im starken Magnetfeld der Falle kann diese Nadel nur zwei Stellungen einnehmen: parallel oder antiparallel zum Feld. Genau diese beiden Stellungen bilden das Qubit [1]:

- **Zustand 0:** Elektronenspin „oben“,
- **Zustand 1:** Elektronenspin „unten“.

Ein Rechenschritt auf einem einzelnen Qubit ist also nichts anderes als das gezielte, teilweise oder vollständige Umklappen dieser Kompassnadel, ausgelöst durch einen genau dosierten Laser- oder Mikrowellenpuls. Zum Auslesen leuchtet man mit ultraviolettem Laserlicht auf das Ion. Nur im Zustand 0 streut das Ion dabei Millionen von Lichtteilchen pro Sekunde und leuchtet hell; im Zustand 1 bleibt es dunkel. Eine Kamera „sieht“ so das Rechenergebnis.

Die Kehrseite: Weil das Qubit eine magnetische Kompassnadel ist, reagiert es empfindlich auf kleinste Schwankungen des Magnetfelds. Schon eine Änderung um ein Milliardstel stört die Rechnung. Mit einem Trick – man klappt das Qubit in der Mitte der Wartezeit einmal um, sodass sich langsame Störungen wieder herausmitteln – verlängert sich die Zeit, in der das Qubit seine Quanteneigenschaften behält, von rund zwei Millisekunden auf rund eine Viertelsekunde [1].

## Phononen: Wenn eine Schwingung zum Quantenobjekt wird

Einzelne Qubits reichen nicht. Für einen Computer müssen Qubits miteinander *wechselwirken*: Das Ergebnis des einen soll beeinflussen, was mit dem anderen geschieht. Doch die Elektronen zweier Ionen, die Dutzende Mikrometer auseinander liegen, spüren einander praktisch nicht. Hier kommt der zweite, oft übersehene Teil der Physik ins Spiel: die **Bewegung der Ionen selbst**.

Ein gefangenes Ion sitzt in seiner Falle wie eine Murmel in einer Schüssel. Stösst man es an, schwingt es hin und her – bei ZuriQ einige Millionen Mal pro Sekunde. In der Quantenwelt kann diese Schwingung nicht beliebig stark oder schwach sein. Ihre Energie lässt sich nur in festen Portionen erhöhen oder verringern, ähnlich wie Licht nur in Portionen – den Photonen – ausgesendet wird. Eine solche Energieportion einer mechanischen Schwingung heisst **Phonon**. Kühlt man das Ion mit Lasern so stark, dass es kein einziges Phonon mehr trägt, ist es zwar im „Grundzustand“, aber nicht völlig in Ruhe: Es bleibt über rund 15 Nanometer verschmiert – eine Folge der Quantenmechanik. Im ETH-Experiment gelingt diese Kühlung so gut, dass die Wahrscheinlichkeit, auch nur ein Phonon anzutreffen, unter einem Prozent liegt [1].

Spannend wird es, wenn mehrere Ionen nebeneinander sitzen. Weil alle positiv geladen sind, stossen sie sich ab, und diese Abstossung wirkt wie unsichtbare Federn zwischen ihnen. Bewegt sich ein Ion, spüren es die anderen. Die Ionen schwingen dann nicht mehr einzeln, sondern in gemeinsamen Mustern – den **Eigenmoden**. Man kennt das von einer Gitarrensaite, die in ihrem Grundton oder in Obertönen schwingen kann, oder von einer Trommel, deren Fell in Ringen und Sektoren vibriert.

Die zweite Animation zeigt das für ein Gitter aus 5×5 Ionen, wie es ZuriQ anstrebt. Jedes Ion sitzt in seiner eigenen Falle und schwingt senkrecht zum Chip. Rot bedeutet „gerade oben“, blau „gerade unten“. Im einfachsten Muster schwingen alle 25 Ionen im Gleichtakt. In anderen Mustern kippt das Gitter wie eine Platte, schwingt wie ein Sattel oder benachbarte Ionen bewegen sich gegeneinander. Insgesamt gibt es genau 25 solcher Muster, eines pro Ion, und jedes hat seine eigene, leicht unterschiedliche Frequenz. Am Schluss der Animation wird ein einzelnes Ion in der Mitte angestossen: Die Schwingung bleibt nicht dort, sondern breitet sich wie eine Welle über das Gitter aus und kehrt teilweise zurück. Diese Welle ist nichts anderes als eine Überlagerung der Eigenmoden – und quantenmechanisch betrachtet ein Phonon, das von Ion zu Ion wandert.

![Animation: Eigenmoden und Ausbreitung einer Schwingung in einem 5×5-Gitter aus Ionen.](personal_blog/gitterschwingungen_5x5.gif)

*Animation 2: Gemeinsame Schwingungsmuster (Eigenmoden) von 25 Ionen in einzelnen Mikrofallen. Zum Schluss breitet sich ein Anstoss in der Mitte als Welle über das Gitter aus. Auslenkungen stark vergrössert.*

Daraus folgt die wichtigste Einsicht dieses Beitrags: **Ein Phonon in einem gemeinsamen Schwingungsmuster gehört nicht einem einzelnen Ion, sondern allen beteiligten Ionen gleichzeitig.** Genau das macht die Bewegung zum Datenbus zwischen den Qubits. In ZuriQs Architektur liegen die Fallen einige Dutzend Mikrometer auseinander. Die Kopplung ist dann schwach, wird aber sehr schnell stärker, wenn man die Ionen zusammenschiebt: Halbiert man den Abstand, wird sie achtmal stärker. Weil ZuriQ Ionen über den Chip fahren kann (Animation 1), lässt sich die Kopplung gezielt ein- und ausschalten.

## Wie die Schwingung zwei Qubits verschränkt

Jetzt fehlt noch ein Bindeglied: Wie wird aus der Schwingung der Ionen eine Verbindung zwischen ihren Qubits – also zwischen den Kompassnadeln der Elektronen?

Das Bindeglied ist Licht. Wenn ein Laserpaar den Elektronenspin umklappt, überträgt es dabei auch einen winzigen Stoss auf das ganze Ion – so wie ein Ball, der von einer Wand abprallt, der Wand einen kleinen Impuls gibt. Durch die Wahl der Laserfrequenz bestimmt man, ob dieser Stoss beim Umklappen ein Phonon erzeugt, eines vernichtet oder die Schwingung unberührt lässt. Damit sind **Qubit und Bewegung miteinander gekoppelt**.

Das einfachste Bild dafür stammt von Ignacio Cirac und Peter Zoller aus dem Jahr 1995 [4]. Man stelle sich zwei Ionen vor, beide im Zustand 0, die gemeinsame Schwingung völlig ruhig. Ein Laserpuls auf das erste Ion klappt dessen Spin nur „halb“ um und erzeugt dabei – ebenfalls nur „halb“ – ein Phonon. Das Ion befindet sich jetzt in einer Überlagerung: „nicht umgeklappt und kein Phonon“ zugleich mit „umgeklappt und ein Phonon“. Ein zweiter Puls auf das andere Ion klappt dessen Spin nur dann um, wenn ein Phonon da ist, und schluckt das Phonon dabei. Am Ende ist die Schwingung wieder ruhig, aber die beiden Qubits sind **verschränkt**: Sie sind entweder beide 0 oder beide 1 – und solange niemand misst, ist offen, welches von beiden. Misst man das eine Ion, kennt man sofort das Ergebnis des anderen. Das Phonon hat die Information „das erste Ion hat umgeklappt“ als Bote zum zweiten Ion getragen und ist danach spurlos verschwunden.

In der Praxis verwendet man eine robustere Variante, die Klaus Mølmer und Anders Sørensen 1999 vorgeschlagen haben [5] und mit der bereits im Jahr 2000 vier Beryllium-Ionen miteinander verschränkt wurden [6]. Dabei beleuchtet man beide Ionen gleichzeitig so, dass auf die gemeinsame Schwingung eine Kraft wirkt, deren Richtung vom Zustand der beiden Qubits abhängt. Die Schwingung wird dadurch auf einen kleinen Umweg geschickt und kehrt nach genau berechneter Zeit exakt an ihren Ausgangspunkt zurück. Von der Bewegung bleibt keine Spur. Was bleibt, ist eine Art Erinnerung an den Umweg in den Qubits: eine sogenannte Phase, deren Grösse davon abhängt, ob die beiden Spins gleich oder verschieden stehen. Genau diese Abhängigkeit verknüpft die beiden Qubits – das Ergebnis ist ein **verschränkender Rechenschritt für zwei Qubits**, ein sogenanntes Zwei-Qubit-Gate.

Dieses Verfahren hat zwei bemerkenswerte Eigenschaften. Erstens funktioniert es auch dann, wenn die Ionen nicht perfekt gekühlt sind, denn es zählt nur die Form des Umwegs, nicht der Startpunkt. Zweitens kann man mit der Wahl des Schwingungsmusters bestimmen, welche Ionen miteinander verschränkt werden: Schwingt in einem Muster ein Ion nicht mit, bleibt es aussen vor. Treibt man dagegen das Muster, in dem alle Ionen im Gleichtakt schwingen, verschränkt ein einziger Puls alle beteiligten Ionen auf einmal. In grossen Penning-Fallen mit Hunderten von Ionen wird genau diese Physik genutzt, um alle Ionen gleichzeitig miteinander wechselwirken zu lassen [7].

## Vom Laserpuls zum Programm

Ein Rechenschritt auf einem Quantencomputer ist also kein Bauteil, durch das ein Ion hindurchfliegt, sondern ein **präzise getimter Lichtpuls**: Frequenz, Stärke, Phase und Dauer bestimmen, was geschieht. Aus Einzelschritten, die ein Qubit drehen, und dem verschränkenden Zwei-Qubit-Gate lässt sich jede beliebige Quantenrechnung zusammensetzen – so wie sich jedes klassische Programm aus wenigen Logikgattern zusammensetzen lässt.

Ein Programmlauf auf dem Ionen-Chip sieht dann so aus: Die Ionen werden mit Lasern gekühlt, alle Qubits in den Zustand 0 gebracht, die Ionen an die richtigen Positionen gefahren, die Pulsfolge abgespielt und am Ende die Helligkeit jedes Ions gemessen. Weil jede Messung ein zufälliges Ergebnis liefert, wiederholt man diesen Ablauf tausende Male. Das eigentliche Rechenergebnis ist die Häufigkeitsverteilung – und ein guter Algorithmus sorgt dafür, dass die richtige Antwort darin klar hervorsticht.

## Der Shor-Algorithmus: Warum Quantencomputer die Verschlüsselung herausfordern

Die bekannteste Anwendung eines Quantencomputers ist der **Shor-Algorithmus**, den Peter Shor 1994 vorgestellt hat [8]. Er zerlegt grosse Zahlen in ihre Primfaktoren – also etwa 15 in 3 mal 5 – und zwar für grosse Zahlen unvergleichlich schneller als jedes bekannte Verfahren auf herkömmlichen Computern.

Warum ist das so bedeutsam? Viele Verschlüsselungs- und Signaturverfahren im Internet, allen voran das RSA-Verfahren [9], beruhen auf einer einfachen Asymmetrie: Zwei grosse Primzahlen zu multiplizieren ist leicht, das Produkt wieder in seine Faktoren zu zerlegen ist dagegen extrem schwer. Bei den heute üblichen Schlüssellängen hat das Produkt über 600 Dezimalstellen; selbst die schnellsten Supercomputer würden für die Zerlegung unvorstellbar lange brauchen. Wer die Faktoren kennt, kann jedoch den geheimen Schlüssel berechnen und mitlesen.

Shors Idee ist ein Umweg über einen **Rhythmus**. Man wählt eine beliebige Zahl, etwa 2, und multipliziert sie immer wieder mit sich selbst. Von jedem Ergebnis behält man nur den Rest, der bei Division durch die zu zerlegende Zahl übrig bleibt – so wie man bei einer Uhr nach 12 Uhr wieder bei 1 beginnt. Für die Zahl 15 ergibt sich die Folge 2, 4, 8, 1, 2, 4, 8, 1, … Die Folge wiederholt sich alle vier Schritte. Aus dieser Wiederholungslänge, der **Periode**, kann man mit einer einfachen Rechnung, die jeder Computer in Sekundenbruchteilen erledigt, die Faktoren 3 und 5 bestimmen. Für eine Zahl mit 600 Stellen ist die Periode allerdings astronomisch lang, und die Folge sieht vollkommen zufällig aus. Sie Schritt für Schritt auszurechnen, bis sie sich wiederholt, ist aussichtslos.

Hier kommt der Quantencomputer ins Spiel. Er berechnet die Folge nicht Glied für Glied, sondern in einer einzigen Überlagerung für alle Positionen zugleich. Das allein nützt noch nichts: Eine Messung würde nur einen einzigen, zufälligen Wert liefern. Der entscheidende zweite Schritt heisst **Quanten-Fourier-Transformation**. Man kann sie sich wie ein Ohr vorstellen, das aus einem Geräusch den Grundton heraushört. Genauer: Sie lässt alle Rechenwege so miteinander interferieren, dass sich alles, was nicht zum Rhythmus der Folge passt, gegenseitig auslöscht, und alles, was im Takt ist, sich verstärkt. Nach dieser Interferenz zeigt eine einzige Messung mit hoher Wahrscheinlichkeit einen Wert, aus dem sich die Periode ablesen lässt. Der Quantencomputer liest also nicht alle Ergebnisse aus – das wäre unmöglich –, sondern übersetzt eine gemeinsame Eigenschaft aller Ergebnisse, ihren Rhythmus, in ein einziges, gut messbares Signal.

Mit gefangenen Ionen wurde der Shor-Algorithmus für die Zahl 15 bereits 2016 in Innsbruck in einer Form ausgeführt, die sich prinzipiell auf grössere Zahlen übertragen lässt [10]. Für Zahlen, die für die Verschlüsselung relevant sind, ist der Weg allerdings noch weit. Jeder einzelne Rechenschritt ist mit kleinen Fehlern behaftet, und bei Milliarden von Schritten summieren sie sich. Man braucht deshalb **Quantenfehlerkorrektur**: Viele fehleranfällige Qubits werden zu einem einzigen, robusten „logischen“ Qubit zusammengefasst. 2019 schätzten Craig Gidney und Martin Ekerå, dass für das Knacken eines heute üblichen RSA-Schlüssels rund 20 Millionen fehleranfällige Qubits und acht Stunden Rechenzeit nötig wären [11]. 2025 senkte Gidney die Schätzung auf weniger als eine Million Qubits und weniger als eine Woche Rechenzeit [12] – allerdings unter Annahmen über Fehlerraten und Geschwindigkeit, die heute noch keine Maschine in dieser Grösse erfüllt.

Trotzdem ist die Gefahr ernst zu nehmen. Daten, die heute abgefangen und gespeichert werden, könnten in einigen Jahren entschlüsselt werden – Fachleute sprechen von „harvest now, decrypt later“. Deshalb hat die US-Standardisierungsbehörde NIST 2024 erste Standards für **Post-Quanten-Kryptographie** veröffentlicht: Verschlüsselungsverfahren, deren Sicherheit auf mathematischen Problemen beruht, für die auch Quantencomputer keinen schnellen Lösungsweg kennen [13]. Die Umstellung von Internetprotokollen, Banken und Behörden auf diese Verfahren hat begonnen.

## Fazit: Warum die Penning-Falle spannend ist

Penning-Fallen-Quantencomputer verbinden zwei Welten: die aussergewöhnlich saubere Physik einzelner Atome und die industrielle Fertigung von Mikrochips. Die Kette vom Atom zum Algorithmus lässt sich in wenigen Sätzen zusammenfassen. Die Information steckt in der Stellung eines einzelnen Elektronenspins im Magnetfeld. Elektroden und Magnetfeld halten die Ionen über dem Chip und können sie verschieben. Die elektrische Abstossung verbindet die Bewegungen der Ionen zu gemeinsamen Schwingungen, deren Energieportionen – die Phononen – allen Ionen gemeinsam gehören. Laserpulse koppeln die Spins an diese Schwingungen und verschränken so Ionen, die sich sonst nicht spüren würden. Aus vielen solchen Schritten entstehen Algorithmen wie der von Shor.

Offen sind vor allem technische Fragen: Wie schnell und fehlerarm lassen sich Zwei-Qubit-Gates zwischen Ionen in getrennten Mikrofallen ausführen? Wie stabil muss das Magnetfeld sein? Und wie wächst man von neun auf Tausende von Ionen, mit der Präzision, die Fehlerkorrektur verlangt? Dass ein Zürcher Start-up diese Fragen mit einer Architektur angeht, die von Grund auf zweidimensional gedacht ist, macht die Entwicklung auch aus Schweizer Sicht besonders interessant.

## Zum Weiterlesen: das ausführliche Skript

Wer tiefer einsteigen möchte, findet im begleitenden **[Skript „Quantencomputer mit gefangenen Ionen in Penning-Fallen“](/personal_blog/quantencomputer_penning_traps_WUCH.pdf)** die vollständigen Herleitungen mit vielen Rechenbeispielen: die Energieniveaus des Beryllium-Ions im starken Magnetfeld, die Berechnung der Schwingungsmuster von drei gekoppelten Ionen, die Funktionsweise des Mølmer-Sørensen-Gates, die Quanten-Fourier-Transformation und den Shor-Algorithmus Schritt für Schritt – inklusive Python-Programmen zum Nachrechnen. Die Programme, mit denen die beiden Animationen in diesem Beitrag erstellt wurden, liegen ebenfalls bei.

---

## Referenzen

[1] S. Jain *et al.*, “Penning micro-trap for quantum computing,” *Nature*, vol. 627, pp. 510–514, 2024, doi: 10.1038/s41586-024-07111-x.

[2] S. Jain, J. Alonso, M. Grau, and J. P. Home, “Scalable arrays of micro-Penning traps for quantum computing and simulation,” *Phys. Rev. X*, vol. 10, no. 3, Art. no. 031027, 2020, doi: 10.1103/PhysRevX.10.031027.

[3] ZuriQ, “ZuriQ raises $25.5 million to scale new groundbreaking 2D quantum architecture,” press release, Jul. 28, 2026. [Online]. Available: https://zuriq.com/fundraise/zuriq-raises-25-5-million/

[4] J. I. Cirac and P. Zoller, “Quantum computations with cold trapped ions,” *Phys. Rev. Lett.*, vol. 74, no. 20, pp. 4091–4094, 1995, doi: 10.1103/PhysRevLett.74.4091.

[5] K. Mølmer and A. Sørensen, “Multiparticle entanglement of hot trapped ions,” *Phys. Rev. Lett.*, vol. 82, no. 9, pp. 1835–1838, 1999, doi: 10.1103/PhysRevLett.82.1835.

[6] C. A. Sackett *et al.*, “Experimental entanglement of four particles,” *Nature*, vol. 404, pp. 256–259, 2000, doi: 10.1038/35005011.

[7] J. G. Bohnet *et al.*, “Quantum spin dynamics and entanglemSent generation with hundreds of trapped ions,” *Science*, vol. 352, no. 6291, pp. 1297–1301, 2016, doi: 10.1126/science.aad9958.

[8] P. W. Shor, “Polynomial-time algorithms for prime factorization and discrete logarithms on a quantum computer,” *SIAM J. Comput.*, vol. 26, no. 5, pp. 1484–1509, 1997, doi: 10.1137/S0097539795293172.

[9] R. L. Rivest, A. Shamir, and L. Adleman, “A method for obtaining digital signatures and public-key cryptosystems,” *Commun. ACM*, vol. 21, no. 2, pp. 120–126, 1978, doi: 10.1145/359340.359342.

[10] T. Monz *et al.*, “Realization of a scalable Shor algorithm,” *Science*, vol. 351, no. 6277, pp. 1068–1070, 2016, doi: 10.1126/science.aad9480.

[11] C. Gidney and M. Ekerå, “How to factor 2048 bit RSA integers in 8 hours using 20 million noisy qubits,” *Quantum*, vol. 5, Art. no. 433, 2021, doi: 10.22331/q-2021-04-15-433.

[12] C. Gidney, “How to factor 2048 bit RSA integers with less than a million noisy qubits,” arXiv:2505.15917, 2025.

[13] National Institute of Standards and Technology, “Module-lattice-based key-encapsulation mechanism standard,” FIPS 203, Aug. 2024, doi: 10.6028/NIST.FIPS.203.
