#importar bibliotecas
import numpy as np
import random
#crear tablero: como variable
#clrtablero=np(10,10," ")
#print(tablero)
#como nos hace falta la función crear tablero mejor no crear la variable si no la función (con tablero dentro)
def crea_tablero(lado=10):
    tablero=np.full((lado,lado)," " )
    return tablero  
tablero= crea_tablero(10)
#pasamos a comentar para no imrpmir tablero inicial y que se solape con los siguientes 
#print(tablero)

# Crear variable barco y funcion Colocar barco: Posiciona un par de barcos en [(0,1), (1,1)] y [(1,3), (1,4), (1,5), (1,6)]./
# Los barcos serán Os mayúsculas. Como ves, un barco de dos posiciones de eslora y otro de cuatro.
#primero creamos barco 1
barco1= [(0,1),(1,1)]
#crear función Colocar barco:
def coloca_barco(tablero,barco):
    for pieza in barco:
        tablero[pieza]="0"
    return tablero
#invoco a la función coloca barco pasandole el tablero
#tablero=coloca_barco(tablero,barco1)
#ahora creo y coloco barco2:
barco2= [(1,3),(1,4),(1,5),(1,6)]
#tablero=coloca_barco(tablero,barco2)
#print(tablero)

#Creamos una función que recompensa el buen tiro del jugador. Por ejemplo si aciertas 2 veces reparas una pieza de tu barco o tienes un tiro extra si no hay pieza que reparar:
#primero creamos la función arregla pieza de barco como arregla barco
def arreglabarco(barco):
    coordenadareparar=tuple(map(int,input("introduce la cooordenada que quieres reparar").split(',')))
    tablero[coordenadareparar]= "0"
    print(tablero)
    return tablero
#necesitamos crear un cambio de turno
jugador0= usuario
jugador1= usuario2

#ahora creamos input de 1 o 2 para que el user elija una recompensa
def eleccionrecompensa(tablero):
    pregunta= (input("Elige 1 si quieres un nuevo tiro y Elige 2 si quieres reparar una pieza de barco"))
    if pregunta == 1:
        recibe_disparo(tablero)
    elif pregunta ==2:
        arreglabarco(tablero)
        cambioturno
    else:
        print("Has introducido un valor no válido pierdes la recompensa")
    return tablero


#FORMAS MAS AUTOMATICA DE CREAR TABLERO,BARCOS Y COLOCARLOS.Nota: entender que barco1,barco2,etc en in es el equivalente a una flota o lista de barcos,[] :
for barco in [barco1,barco2]:
    tablero = coloca_barco(tablero,barco)
    #si dejo este print imprimo cada vez que meto un barco y los acumula: print(tablero) print(f"Coloco:{barco}",)
print(tablero)
#CREAR FUNCION DISPARO. 3. Recibe un disparo en uno de los barcos, sustituyendo la O por una X
def recibe_disparo(tablero):
    #for disparo in tablero creamos un input de coordenada:
    control = False
    contador=0
    while not control:
        coordenadarecib = tuple(map(int,input("PC introduce la cooordenada").split(',')))
        if tablero[coordenadarecib]=="0": 
            tablero[coordenadarecib]="X"
            print(f" Tocado")
            print(tablero)
            control= False 
            contador=contador+1
            if contador==2:
                eleccionrecompensa(tablero)
                #print(f" Contador: {contador}")
                contador=0
                print(contador)
        elif tablero[coordenadarecib]=="X":
            print("Dispara a otro punto más tarde,ahora cambia de turno")
            control= True
        else:
            print(f"Agua, cambia de turno")
            control= True
    return tablero
    #else:
    #    print("Cambio de turno")

tablero=recibe_disparo(tablero)

print(tablero)

#Funcion para comprobar que barco aleatorio cabe en el tablero_
def barcoapto(tablero,barco):
    tablerotemp= tablero.copy()
    filasmax = tablero.shape[0]
    columnasmax = tablero.shape[1]
    for pieza in barco:
        fila=pieza[0]
        columna=pieza[1]
        if fila <0 or fila>= filasmax:
            print(f" No puedo poner {pieza} porque sale del tablero")
            return False
        if columna <0 or columna>= columnasmax:
            print(f" No puedo poner {pieza} porque sale del tablero")
            return False
        if tablero[pieza] == "X" or tablero[pieza] == "0":
            print(f" No puedo poner {pieza} porque hay otro barco")
            return False
        tablero_temp[pieza]= "0"
    return tablerotemp

#Ahora podemos crear barcos aleatorios a los que aplicar la función de comprobación barco apto

def crea_barco_aleatorio(tablero,eslora = 4, num_intentos = 100):
    num_max_filas = tablero.shape[0]
    num_max_columnas = tablero.shape[1]
    while True:
        barco = []
        # Construimos el hipotetico barco
        pieza_original = (random.randint(0,num_max_filas-1),random.randint(0, num_max_columnas -1))
        print("Pieza original:", pieza_original)
        barco.append(pieza_original)
        orientacion = random.choice(["N","S","O","E"])
        print("Con orientacion", orientacion)
        fila = pieza_original[0]
        columna = pieza_original[1]
        for i in range(eslora -1):
            if orientacion == "N":
                fila -= 1
            elif orientacion  == "S":
                fila += 1
            elif orientacion == "E":
                columna += 1
            else:
                columna -= 1
            pieza = (fila,columna)
            barco.append(pieza)
        tablero_temp = coloca_barco_plus(tablero, barco)
        if type(tablero_temp) == np.ndarray:
            return tablero_temp
        print("Tengo que intentar colocar otro barco")