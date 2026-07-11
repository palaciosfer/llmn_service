# API de Diagnostico Agricola

Microservicio que recibe el diagnostico de la CNN (ejecutada en el movil) junto con el texto de sintomas del usuario, y devuelve tratamiento, prevencion y fuentes usando RAG (busqueda hibrida TF-IDF + BERT + LLM Qwen).

**Base URL en produccion:** `http://52.1.110.21:8000`

**Documentacion interactiva (Swagger):** `http://52.1.110.21:8000/docs`

---

## Autenticacion (JWT)

Todos los endpoints bajo `/api/v1/` requieren un token JWT en el header `Authorization`.

El token lo emite el **microservicio de usuarios**. El `JWT_SECRET` debe ser el mismo en ambos microservicios.

### Header requerido

```
Authorization: Bearer <token_jwt>
```

### Payload esperado del JWT

```json
{
  "sub": "id-del-usuario",
  "rol": "agricultor",
  "email": "usuario@ejemplo.com",
  "iat": 1700000000,
  "exp": 1700086400
}
```

| Campo | Tipo     | Requerido | Descripcion                                |
|-------|----------|-----------|--------------------------------------------|
| `sub` | `string` | si        | ID unico del usuario                      |
| `rol` | `string` | si        | `"agricultor"` o `"aprendiz"`              |
| `email`| `string`| no        | Email del usuario                          |
| `iat` | `number` | si        | Timestamp de emision (epoch seconds)       |
| `exp` | `number` | si        | Timestamp de expiracion (epoch seconds)    |

### Errores de autenticacion

| HTTP | Cuerpo                                    | Causa                          |
|------|-------------------------------------------|--------------------------------|
| 401  | `{"detail": "Not authenticated"}`         | No se envio el header Bearer   |
| 401  | `{"detail": "Token expirado"}`            | El token expiro                |
| 401  | `{"detail": "Token invalido"}`            | Firma incorrecta o malformado  |
| 403  | `{"detail": "Rol no valido: ..."}`        | Rol no es agricultor/aprendiz  |

### Token de desarrollo (solo con DEV_MODE=true)

Para pruebas sin el microservicio de usuarios:

```
POST /api/v1/dev/token
Content-Type: application/json

{
  "sub": "test-user-1",
  "rol": "agricultor",
  "email": "test@ejemplo.com"
}
```

**Response 200:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "rol": "agricultor",
  "expira_en_horas": 24
}
```

---

## Endpoints

### 1. `POST /api/v1/consultar` — Endpoint principal (movil)

Este es el endpoint que usa la app movil. Recibe el resultado de la CNN (que corre local en el telefono) mas el texto que escribe el usuario, y devuelve el diagnostico completo con tratamiento.

**Headers:**
```
Authorization: Bearer <token_jwt>
Content-Type: application/json
```

**Request body:**

```json
{
  "resultado_cnn": {
    "cultivo": "calabaza",
    "enfermedad": "oidio",
    "confianza": 0.85,
    "clase_cnn": "Calabaza_Powdery Mildew",
    "confianza_baja": false
  },
  "texto": "las hojas tienen un polvo blanco y se ven manchas",
  "cultivos": ["calabaza"]
}
```

| Campo               | Tipo     | Requerido | Descripcion                                            |
|----------------------|----------|-----------|--------------------------------------------------------|
| `resultado_cnn`      | `object` | **si**    | Resultado de la CNN ejecutada en el movil              |
| `resultado_cnn.cultivo` | `string` | **si** | Cultivo detectado (ej: `"calabaza"`, `"tomate"`)       |
| `resultado_cnn.enfermedad` | `string` | **si** | Enfermedad detectada (ej: `"oidio"`, `"late blight"`) |
| `resultado_cnn.confianza` | `number` | **si** | Confianza de la prediccion, entre `0.0` y `1.0`       |
| `resultado_cnn.clase_cnn` | `string` | **si** | Etiqueta cruda del modelo (ej: `"Calabaza_Powdery Mildew"`) |
| `resultado_cnn.confianza_baja` | `boolean` | **si** | `true` si la confianza es menor al umbral (0.50)  |
| `texto`              | `string` | no        | Sintomas que describe el usuario. Max 2000 caracteres  |
| `cultivos`           | `string[]` | no      | Lista de cultivos para filtrar la busqueda. Si esta vacio, no filtra |

**Response 200:**

```json
{
  "modo": "online",
  "diagnostico": {
    "cultivo": "calabaza",
    "enfermedad": "oidio",
    "confianza_original": 0.85,
    "confianza_ajustada": 0.99,
    "estado": "reforzado",
    "sintomas_refuerzo": ["cenicilla", "oidio", "polvo blanco"],
    "sintomas_contradiccion": [],
    "consulta": "calabaza oidio cenicilla polvo blanco manchas lesiones hojas",
    "explicacion": "El texto refuerza el diagnostico (coincide en: cenicilla, oidio, polvo blanco). Confianza 0.85 -> 0.99."
  },
  "sintomas": ["oidio", "cenicilla", "polvo blanco", "manchas", "lesiones", "hojas"],
  "avisos": [],
  "n_documentos": 10,
  "respuesta": {
    "texto": "DIAGNOSTICO:\nLa planta tiene oidio (cenicilla)...\n\nTRATAMIENTO:\nAplicar azufre elemental 80% PM...\n\nPREVENCION:\nUsar variedades resistentes...",
    "diagnostico": "La planta tiene oidio (cenicilla)...",
    "tratamiento": "Aplicar azufre elemental 80% PM...",
    "prevencion": "Usar variedades resistentes al oidio...",
    "fuentes": [
      "Guia GIP Cucurbitaceas - MAPA Espana 2023",
      "Produccion de Calabaza - UPR Mayaguez"
    ],
    "rol": "agricultor",
    "sin_documentos": false
  }
}
```

| Campo de response     | Tipo       | Descripcion                                             |
|------------------------|------------|---------------------------------------------------------|
| `modo`                 | `string`   | `"online"` (siempre en servidor)                       |
| `diagnostico`          | `object`   | Resultado de la fusion CNN + NLP                        |
| `diagnostico.cultivo`  | `string`   | Cultivo detectado                                       |
| `diagnostico.enfermedad` | `string` | Enfermedad detectada                                    |
| `diagnostico.confianza_original` | `number` | Confianza de la CNN original                   |
| `diagnostico.confianza_ajustada` | `number` | Confianza ajustada tras analizar el texto      |
| `diagnostico.estado`   | `string`   | `"reforzado"`, `"posible_contradiccion"` o `"sin_senal_textual"` |
| `diagnostico.explicacion` | `string` | Explicacion legible del ajuste de confianza           |
| `sintomas`             | `string[]` | Sintomas extraidos del texto del usuario                |
| `avisos`               | `string[]` | Advertencias (confianza baja, cultivo no registrado)    |
| `n_documentos`         | `number`   | Cantidad de documentos usados para la respuesta         |
| `respuesta.texto`      | `string`   | Respuesta completa generada por el LLM                  |
| `respuesta.diagnostico`| `string`   | Seccion de diagnostico extraida                         |
| `respuesta.tratamiento`| `string`   | Seccion de tratamiento extraida                         |
| `respuesta.prevencion` | `string`   | Seccion de prevencion extraida                          |
| `respuesta.fuentes`    | `string[]` | Fuentes bibliograficas de los documentos usados         |
| `respuesta.rol`        | `string`   | Rol usado para generar la respuesta                     |
| `respuesta.sin_documentos` | `boolean` | `true` si no se encontraron documentos relevantes    |

**Diferencia por rol:**

| Rol          | Tipo de respuesta                                        |
|--------------|----------------------------------------------------------|
| `agricultor` | Lenguaje simple, directo, practico. Sin tecnicismos.     |
| `aprendiz`   | Tecnica y educativa. Menciona agente causal y porques.   |

El rol se toma automaticamente del JWT, no se envia en el body.

---

### 2. `GET /health` — Liveness

No requiere autenticacion.

```
GET /health
```

**Response 200:**
```json
{
  "status": "ok",
  "version": "1.0.0"
}
```

---

### 3. `GET /ready` — Readiness

No requiere autenticacion. Devuelve 200 si todo esta listo, 503 si algo falta.

```
GET /ready
```

**Response 200:**
```json
{
  "status": "ready",
  "cnn_disponible": true,
  "bert_disponible": true,
  "ollama_disponible": true,
  "modulos_cargados": true
}
```

**Response 503** (algo no esta listo):
```json
{
  "detail": {
    "status": "degraded",
    "cnn_disponible": true,
    "bert_disponible": true,
    "ollama_disponible": false,
    "modulos_cargados": true
  }
}
```

---

### 4. `POST /api/v1/diagnosticar` — Diagnostico con imagen

Alternativa que envia la imagen al servidor (la CNN corre en el servidor). Usa `multipart/form-data`.

**Headers:**
```
Authorization: Bearer <token_jwt>
Content-Type: multipart/form-data
```

| Campo      | Tipo     | Requerido | Descripcion                                    |
|------------|----------|-----------|------------------------------------------------|
| `imagen`   | `file`   | **si**    | Imagen de la planta (JPEG, PNG, WebP, BMP, TIFF). Max 8 MB |
| `texto`    | `string` | no        | Sintomas observados                            |
| `cultivos` | `string` | no        | Cultivos separados por coma: `"tomate,calabaza"` |

**Response 200:** mismo formato que `/api/v1/consultar`.

**Errores especificos:**

| HTTP | Causa                                    |
|------|------------------------------------------|
| 400  | Imagen vacia o excede 8 MB               |
| 415  | Tipo de archivo no soportado             |
| 503  | CNN no cargada o Ollama no disponible    |
| 504  | Timeout en la generacion del LLM         |

---

### 5. `POST /api/v1/embeddings` — Generar embeddings

**Headers:**
```
Authorization: Bearer <token_jwt>
Content-Type: application/json
```

```json
{
  "textos": ["hoja con manchas negras", "polvo blanco en calabaza"]
}
```

**Response 200:**
```json
{
  "embeddings": [[0.012, -0.034, ...], [0.045, 0.021, ...]],
  "dimension": 384
}
```

---

### 6. `POST /api/v1/generar` — Generar texto con LLM

Llamada directa al LLM sin el pipeline RAG. Util para pruebas.

**Headers:**
```
Authorization: Bearer <token_jwt>
Content-Type: application/json
```

```json
{
  "prompt": "Que es el oidio en calabaza?",
  "temperatura": 0.2
}
```

**Response 200:**
```json
{
  "texto": "El oidio es una enfermedad fungica..."
}
```

---

## Codigos de error comunes

| HTTP | Codigo                  | Significado                                   |
|------|-------------------------|-----------------------------------------------|
| 400  | `IMAGEN_INVALIDA`       | No se pudo decodificar la imagen              |
| 401  | —                       | Token JWT faltante, expirado o invalido       |
| 403  | —                       | Rol no autorizado                              |
| 415  | —                       | Tipo de imagen no soportado                    |
| 503  | `MODELO_NO_DISPONIBLE`  | Un modelo (CNN/BERT) no esta cargado          |
| 503  | `OLLAMA_NO_DISPONIBLE`  | Ollama no responde                             |
| 504  | `TIMEOUT_INFERENCIA`    | Ollama tardo mas del timeout configurado       |

Formato de error:
```json
{
  "detail": "Descripcion del error en espanol"
}
```

---

## Ejemplo completo desde movil (pseudocodigo)

```javascript
// 1. El usuario ya tiene su JWT del login
const token = "eyJhbGciOiJIUzI1NiIs...";

// 2. La CNN corre local en el telefono y da el resultado
const resultadoCnn = {
  cultivo: "tomate",
  enfermedad: "late blight",
  confianza: 0.72,
  clase_cnn: "Tomato___Late_blight",
  confianza_baja: false
};

// 3. El usuario escribe los sintomas
const textoUsuario = "manchas oscuras en las hojas y humedad";

// 4. Llamada al servidor
const response = await fetch("http://52.1.110.21:8000/api/v1/consultar", {
  method: "POST",
  headers: {
    "Authorization": `Bearer ${token}`,
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    resultado_cnn: resultadoCnn,
    texto: textoUsuario,
    cultivos: ["tomate"]
  })
});

if (response.status === 401) {
  // Token expirado -> renovar token con el microservicio de usuarios
}

if (response.status === 503) {
  // Servidor no disponible -> mostrar mensaje al usuario
}

const data = await response.json();

// 5. Mostrar al usuario
console.log(data.respuesta.tratamiento);  // "Aplicar fungicida..."
console.log(data.respuesta.prevencion);   // "Rotar cultivos..."
console.log(data.respuesta.fuentes);      // ["Guia GIP...", ...]
console.log(data.diagnostico.confianza_ajustada); // 0.87
```

---

## Variables de entorno (.env)

| Variable        | Requerida | Ejemplo                        | Descripcion                                    |
|-----------------|-----------|--------------------------------|------------------------------------------------|
| `JWT_SECRET`    | **si**    | `mi-clave-secreta-produccion`  | Debe coincidir con el microservicio de usuarios |
| `JWT_ALGORITHM` | no        | `HS256`                        | Algoritmo JWT (default: HS256)                 |
| `QWEN_MODELO`   | no        | `qwen3.5:4b`                  | Modelo LLM en Ollama (default: qwen3.5:0.8b)  |
| `OLLAMA_TIMEOUT`| no        | `180`                          | Timeout de Ollama en segundos (default: 120)   |
| `DEV_MODE`      | no        | `true`                         | Habilita `/api/v1/dev/token` (default: false)  |
| `CORS_ORIGINS`  | no        | `*`                            | Origenes permitidos (default: *)               |
| `LOG_LEVEL`     | no        | `INFO`                         | Nivel de logging (default: INFO)               |
| `MAX_IMAGEN_MB` | no        | `8`                            | Limite de imagen en MB (default: 8)            |

---

## Despliegue en EC2

```bash
# 1. Crear el archivo .env en el servidor
nano .env

# 2. Ejecutar el script de despliegue
chmod +x deploy.sh
./deploy.sh
```

El script instala Docker, levanta la API y Ollama, descarga el modelo LLM, y deja todo funcionando.
