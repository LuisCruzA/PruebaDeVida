// Array para guardar los registros de las pruebas
const registros = [];

// Función para agregar un registro
function agregarRegistro(nombreImagen, resultado) {
    const registro = {
        imagen: nombreImagen,
        resultado: resultado,
        fecha: new Date().toLocaleString()
    };
    registros.push(registro);
    console.log("Registro agregado:", registro);
    mostrarHistorial();
}

// Función para mostrar el historial en consola
function mostrarHistorial() {
    console.log("----- Historial de registros -----");
    registros.forEach((registro, index) => {
        console.log(
            `${index + 1}. Imagen: ${registro.imagen} | Resultado: ${registro.resultado} | Fecha: ${registro.fecha}`
        );
    });
}

// Función que se ejecuta al hacer click en el botón
function verificarCumplimiento() {
    const resultado = Math.random() < 0.5 ? "Cumple" : "No cumple";
    const nombreImagen = document.getElementById("randomImage").src; // obtiene el src de la imagen actual
    agregarRegistro(nombreImagen, resultado);

    const resultadoElem = document.getElementById("resultado");
    resultadoElem.textContent = resultado;
    resultadoElem.style.color = resultado === "Cumple" ? "green" : "red";
}

// Conectar la función al botón
document.getElementById("verificarBtn").addEventListener("click", verificarCumplimiento);
