class Matrise:
    """
    Attributter:
      - A: En nested liste av flyttall med dimensjon m×n.
      - m: Antall rader i matrisen.
      - n: Antall kolonner i matrisen.
    
    Indeksering:
      - bruker 1-indeksering for å samsvare med standard matte notasjon av matriser
      - i er radindeks (1-basert: 1, 2, ..., m).
      - j er kolonneindeks (1-basert: 1, 2, ..., n).
    """

    def __init__(self, A: list[list[float]]):
        """
        Oppretter Matrise fra en liste av lister (A).
        """
        self.m = len(A)
        self.n = len(A[0])
        self.A = [row[:] for row in A]

    def Skalarmultiplikasjon(self, k: float) -> "Matrise":
        """
        Ganger alle elementene i matrisen med skalar k.
        (k·A)_{ij} = k * A_{ij}.
        """
        return Matrise([
            [k * a_ij 
            for a_ij in row] 
            for row in self.A
        ])

    def Matrisemultiplikasjon(self, B: "Matrise") -> "Matrise":
        """
        Utfører vanlig matrisemultiplikasjon:
        (A×B)_{ij} = ∑ A_{ik}·B_{kj}.
        (bruker 0-indeksering for penere kode her)
        """
        return Matrise([
            [sum(self.A[i][k] * B.A[k][j] for k in range(self.n)) # regner ut skalarproduktet
            for j in range(B.n)] # Itererer over alle kolonner i B (og dermed i produktmatrisen)
            for i in range(self.m) # Itererer over alle rader i self (og dermed i produktmatrisen)
            
        ])

    def MatrisePotens(self, n):
        """
        Beregner den n-te potensen av matrisen rekursivt ved matrisemultiplikasjon med seg selv
        """
        if n == 1: # base case: A^1
            return self

        return self.Matriseaddisjon(self.MatrisePotens(n-1))

    def Matriseaddisjon(self, B: "Matrise") -> "Matrise":
        """
        Utfører elementvis addisjon av to matriser.
        """
        return Matrise([
            [self.A[i][j] + B.A[i][j] 
            for j in range(self.n)] 
            for i in range(self.m)
        ])

    def Determinant(self) -> float:
        """
        Bruker Laplace-ekspansjon for å rekursivt finne determinant:
        det(A) = ∑ (-1)^(1+j)*A_{1j}*Minor(1,j).
        """
        if self.m == 1: # base case: 1×1-matrise
            return self.A[0][0]
        
        det = 0
        for j in range(1, self.n + 1): # Laplace-ekspansjon
            det += self.A[0][j-1] * self.Kofaktor(1, j)
        return det
    
    def Minor(self, i: int, j: int) -> "Matrise":
        """
        Returnerer minormatrisen ved å fjerne i-te rad og j-te kolonne.
        """
        return Matrise([
            row[:j-1] + row[j:] 
            for row in (self.A[:i-1] + self.A[i:])
        ])
    
    def Kofaktor(self, i: int, j: int) -> float:
        """
        (-1)^(i+j) * determinant av Minor(i, j).
        """
        return (-1)**(i + j) * self.Minor(i, j).Determinant()

    def Kofaktormatrise(self) -> "Matrise":
        """
        Matrise av samme størrelse der hvert element er Kofaktor(i, j).
        """
        return Matrise([
            [ self.Kofaktor(i, j) 
              for j in range(1, self.n + 1) ]
            for i in range(1, self.m + 1)
        ])
    
    def Transposisjon(self) -> "Matrise":
        """
        Bytter om rader og kolonner: T(A)_{ij} = A_{ji}.
        (bruker 0-indeksering for penere kode her)
        """
        return Matrise([
            [self.A[i][j] 
            for i in range(self.m)]
            for j in range(self.n)
        ])
    
    def Adjungert(self) -> "Matrise":
        """
        Transposisjonen av Kofaktormatrise(A).
        """
        return self.Kofaktormatrise().Transposisjon()
    
    def Inversmatrise(self) -> "Matrise":
        """
        A^-1 = (1/det(A)) * Adj(A), for det(A) ≠ 0.
        """
        return self.Adjungert().Skalarmultiplikasjon(1 / self.Determinant())

    # Radoprasjoner
    def Radbytte(self, p, q) -> "Matrise":
        """
        Returnerer en ny matrise der rad p og rad q er byttet.
        R_p <--> R_q
        """
        return Matrise([
            self.A[q-1][:] if i == p-1 else self.A[p-1][:] if i == q-1 else self.A[i][:]
            for i in range(self.m)
        ])

    def RadSkaler(self, p: int, k: float) -> "Matrise":
        """
        Returnerer en ny matrise der rad p er multiplisert med k.
        R_p --> k * R_p
        """
        return Matrise([
            [k * self.A[p-1][j] for j in range(self.n)] if i == p-1 else self.A[i][:]
            for i in range(self.m)
        ])

    def RadAddisjon(self, p: int, q: int, k: float) -> "Matrise":
        """
        Returnerer en ny matrise der k ganger rad q legges til rad p.
        R_p --> R_p + k * R_q
        """
        return Matrise([
            [self.A[p-1][j] + k*self.A[q-1][j] for j in range(self.n)]
            if i == p-1 else self.A[i][:]
            for i in range(self.n)
        ])


    def __str__(self, NumLen=5, dec=1):
        """
        Returnerer en pen strengrepresentasjon av matrisen med en grafisk ramme.

        Parametere:
        NumLen (int): Antall tegn for hvert tall (bredde) i utskriften.
        dec (int): Antall desimaler som skal vises for hvert tall.
        """
        return '\n'.join(
            ["┌" + " " * NumLen * len(self.A[0]) + "  ┐"] +
            ["|" + "".join(f"{a_ij:>{NumLen}.{dec}f}" for a_ij in row) + "  |" for row in self.A] +
            ["└" + " " * NumLen * len(self.A[0]) + "  ┘"])


class UtvidetMatrise(Matrise):
    """
    Attributter:
      - A: En nested liste av flyttall med dimensjon m×n (hoveddelen av matrisen).
      - B: En nested liste av flyttall med dimensjon m×k (augmented delen av matrisen).
      - m: Antall rader i matrisen.
      - n: Antall kolonner i hovedmatrisen A.
      - k: Antall kolonner i den augmented delen B.
    """

    def __init__(self, A: list[list[float]], B: list[list[float]]):
        super().__init__(A)
        self.B = [row[:] for row in B]
        self.k = len(B[0])

    def Radbytte(self, p, q) -> "UtvidetMatrise":
        super().Radbytte(p, q)  # Bytt rader i A
        # Bytt rader i B
        self.B[p-1], self.B[q-1] = self.B[q-1], self.B[p-1]
        return self
    
    def __str__(self, NumLen=5, dec=1):
        """
        Returnerer en pen strengrepresentasjon av en augmented matrix med en grafisk ramme.

        Metoden formaterer to blokker:
            - Venstre blokk (self.A): Hoveddelen av matrisen.
            - Høyre blokk (self.B): Augmented delen.
      
        Parametere:
        NumLen (int): Antall tegn for hvert tall (bredde) i utskriften.
        dec (int): Antall desimaler som skal vises for hvert tall.
        """
        return '\n'.join(
            ["┌" + " " *NumLen*(len(self.A[0])+len(self.B[0]))+ "     ┐"] +
            ["|" + "".join(f"{a_ij:>{NumLen}.{dec}f}" for a_ij in self.A[i]) + "  |" + "".join(f"{a_ij:>{NumLen}.{dec}f}" for a_ij in self.B[i]) + "  |" for i in range(self.m)] +
            ["└" + " " *NumLen*(len(self.A[0])+len(self.B[0])) + "     ┘"])



def Identitetsmatrise(n):
    """
    Lager en identitetsmatrise av dimensjon n×n.
    """
    return Matrise([
        [1 if i==j else 0 
        for j in range(n)]
        for i in range(n)
    ])




