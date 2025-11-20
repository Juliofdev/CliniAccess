import os
import sys

# --- Ajustar path del proyecto para imports ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# --- Importar módulos del proyecto ---
try:
    import capaentrada
    import capaocultaentrenamiento
    import capasalidarecfacial
except Exception as e:
    print("Error importando módulos:", e)
    sys.exit(1)


def mostrar_menu():
    print("\n========== RECONOCIMIENTO FACIAL ==========\n")
    print("1) Capturar rostros (capaentrada)")
    print("2) Entrenar modelo (capaprocesar)")
    print("3) Ejecutar reconocimiento (capasalida)")
    print("4) Salir\n")


def main():
    while True:
        mostrar_menu()
        opcion = input("Selecciona una opción: ")

        if opcion == "1":
            print("\n▶ Iniciando captura de rostros...\n")
            capaentrada.iniciar()

        elif opcion == "2":
            print("\n▶ Iniciando entrenamiento...\n")
            capaocultaentrenamiento.entrenar()

        elif opcion == "3":
            print("\n▶ Ejecutando reconocimiento facial...\n")
            capasalidarecfacial.iniciar()

        elif opcion == "4":
            print("\n Saliendo del programa.")
            break

        else:
            print("Opción inválida, intenta nuevamente.")


if __name__ == "__main__":
    main()
