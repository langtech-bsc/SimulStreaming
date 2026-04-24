import torch
from transformers import WhisperForConditionalGeneration
import os
from collections import OrderedDict

# --- Configuración ---
MODEL_PATH_FRAGMENTED = "./whisper-large-v3-tiny-caesar" 
OUTPUT_PATH = "./whisper-large-v3-tiny-pt" # Nueva versión, FINAL_8
OUTPUT_MODEL_FILENAME = "large-v3.pt"
# --- Fin de Configuración ---

# Crear el directorio de salida
os.makedirs(OUTPUT_PATH, exist_ok=True)
output_model_file = os.path.join(OUTPUT_PATH, OUTPUT_MODEL_FILENAME)

print("⏳ 1. Cargando el modelo fragmentado (Hugging Face) desde la ruta especificada...")
try:
    # 1. Cargar el modelo de Hugging Face
    model_hf = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH_FRAGMENTED)
    config = model_hf.config 
    state_dict_hf = model_hf.state_dict()
    
    # -------------------------------------------------------------------
    print("⏳ 2. Aplicando correcciones de claves críticas (LayerNorm del Decoder)...")

    # Mapeo de la clave del embedding (Corrección del error anterior, garantizada con try/except)
    try:
        # Intenta variante 1: sin prefijo 'model.'
        state_dict_hf["decoder.token_embedding.weight"] = state_dict_hf.pop("decoder.embed_tokens.weight")
    except KeyError:
        # Intenta variante 2: con prefijo 'model.'
        try:
            state_dict_hf["decoder.token_embedding.weight"] = state_dict_hf.pop("model.decoder.embed_tokens.weight")
        except KeyError:
             pass # La clave ya está en el formato correcto, no hace falta renombrar.
            
    # Mapeo de la clave de LayerNorm del Decoder (SOLUCIÓN PARA TU ÚLTIMO ERROR)
    # Intentamos renombrar la clave con y sin prefijo "model."
    
    # 1. Intentamos la variante A (Sin prefijo 'model.')
    if "decoder.layer_norm.weight" in state_dict_hf:
        print("  -> Renombrando 'decoder.layer_norm.weight' a 'decoder.ln.weight'...")
        state_dict_hf["decoder.ln.weight"] = state_dict_hf.pop("decoder.layer_norm.weight")
    if "decoder.layer_norm.bias" in state_dict_hf:
        state_dict_hf["decoder.ln.bias"] = state_dict_hf.pop("decoder.layer_norm.bias")

    # 2. Intentamos la variante B (Con prefijo 'model.'), si las anteriores no existían
    if "model.decoder.layer_norm.weight" in state_dict_hf:
        print("  -> Renombrando 'model.decoder.layer_norm.weight' a 'decoder.ln.weight'...")
        state_dict_hf["decoder.ln.weight"] = state_dict_hf.pop("model.decoder.layer_norm.weight")
    if "model.decoder.layer_norm.bias" in state_dict_hf:
        state_dict_hf["decoder.ln.bias"] = state_dict_hf.pop("model.decoder.layer_norm.bias")

    # Eliminamos el Unexpected Key restante, si sigue ahí
    if "proj_out.weight" in state_dict_hf:
        del state_dict_hf["proj_out.weight"]

    # -------------------------------------------------------------------
    print("⏳ 3. Mapeando y corrigiendo el resto de claves del state_dict...")

    state_dict_final = OrderedDict()
    
    # Este prefijo será eliminado para el resto de las claves.
    PREFIX_TO_REMOVE = "model." 

    # Mapeo de sub-módulos para corregir la estructura interna (layers -> blocks, k_proj -> key, etc.)
    MAPPING = {
        ".layers.": ".blocks.",
        ".self_attn.k_proj.": ".attn.key.",
        ".self_attn.v_proj.": ".attn.value.",
        ".self_attn.q_proj.": ".attn.query.",
        ".self_attn.out_proj.": ".attn.out.",
        ".self_attn_layer_norm.": ".attn_ln.",
        ".encoder_attn.k_proj.": ".cross_attn.key.",
        ".encoder_attn.v_proj.": ".cross_attn.value.",
        ".encoder_attn.q_proj.": ".cross_attn.query.",
        ".encoder_attn.out_proj.": ".cross_attn.out.",
        ".encoder_attn_layer_norm.": ".cross_attn_ln.",
        ".fc1.": ".mlp.0.",
        ".fc2.": ".mlp.2.",
        ".final_layer_norm.": ".mlp_ln.",
        "encoder.layer_norm.": "encoder.ln_post.",
        # La línea "decoder.layer_norm.": "decoder.ln." se elimina de aquí ya que se maneja arriba con pop
    }
    
    for key, value in state_dict_hf.items():
        new_key = key
        
        # 1. Eliminar el prefijo principal ("model.")
        if new_key.startswith(PREFIX_TO_REMOVE):
            new_key = new_key[len(PREFIX_TO_REMOVE):]
            
        # 2. Aplicar mapeo de sub-módulos
        for hf_sub, openai_sub in MAPPING.items():
            if hf_sub in new_key:
                new_key = new_key.replace(hf_sub, openai_sub)

        # 3. Mapeo del embedding posicional
        if new_key == "encoder.embed_positions.weight":
            new_key = "encoder.positional_embedding"
        elif new_key == "decoder.embed_positions.weight":
            new_key = "decoder.positional_embedding"
            
        state_dict_final[new_key] = value

    # -------------------------------------------------------------------
    print("⏳ 4. Construyendo el diccionario de metadatos (Clave 'dims')...")

    dims = OrderedDict([
        ('n_mels', config.num_mel_bins),
        ('n_vocab', config.vocab_size),
        ('n_audio_ctx', config.max_source_positions),
        ('n_audio_state', config.d_model),
        ('n_audio_head', config.encoder_attention_heads),
        ('n_audio_layer', config.encoder_layers),
        ('n_text_ctx', config.max_target_positions),
        ('n_text_state', config.d_model),
        ('n_text_head', config.decoder_attention_heads),
        ('n_text_layer', config.decoder_layers)
    ])

    # -------------------------------------------------------------------
    print(f"⏳ 5. Guardando el checkpoint completamente corregido en **{output_model_file}**...")
    
    final_checkpoint = {
        "dims": dims,
        "model_state_dict": state_dict_final
    }
    
    torch.save(final_checkpoint, output_model_file)
    
    print("🎉 Éxito: Modelo guardado con el mapeo de claves interno correcto.")
    print(f"Intenta cargar el archivo: {output_model_file}")

except Exception as e:
    print(f"❌ Error crítico en el guardado: {e}")
    print(f"Asegúrate de que el directorio {MODEL_PATH_FRAGMENTED} exista y contenga los archivos del modelo de Hugging Face.")