import re
import asyncio
from textwrap import dedent
import argparse

import outlines
from transformers import AutoTokenizer, logging

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Constantes para el modelo
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"  # Ejemplo, usa el que prefieras
DEVICE = "cpu" # Remove or change to 'cuda'/'mps' if you are using another device.
T_TYPE = "float16" if DEVICE == "cuda" else "float32"

logging.set_verbosity_error()

def format_functions(functions):
    formatted_functions = []
    for func in functions:
        function_info = f"{func['name']}: {func['description']}\n"
        if 'parameters' in func and 'properties' in func['parameters']:
            for arg, details in func['parameters']['properties'].items():
                description = details.get('description', 'No description provided')
                function_info += f"- {arg}: {description}\n"
        formatted_functions.append(function_info)
    return "\n".join(formatted_functions)

SYSTEM_PROMPT_FOR_CHAT_MODEL = dedent("""
    You are an expert designed to call the correct function to solve a problem based on the user's request.
    The functions available (with required parameters) to you are:
    {functions}
    
    You will be given a user prompt and you need to decide which function to call.
    You will then need to format the function call correctly and return it in the correct format.
    The format for the function call is:
    [func1(params_name=params_value)]
    NO other text MUST be included.
                                      
    For example:
    Request: I want to order a cheese pizza from Pizza Hut.
    Response: [order_food(restaurant="Pizza Hut", item="cheese pizza", quantity=1)]
                                      
    Request: Is it raining in NY.
    Response: [get_weather(city="New York")]

    Request: I need a ride to SFO.
    Response: [order_ride(destination="SFO")]
                                      
    Request: I want to send a text to John saying Hello.
    Response: [send_text(to="John", message="Hello!")]
""")

USER_PROMPT_FOR_CHAT_MODEL = dedent("""
    Request: {user_prompt}. 
""")

def instruct_prompt(question, functions, tokenizer):
    messages = [
        {"role": "user", "content": SYSTEM_PROMPT_FOR_CHAT_MODEL.format(functions=format_functions(functions))},
        {"role": "assistant", "content": "I understand and will only return the function call in the correct format."},
        {"role": "user", "content": USER_PROMPT_FOR_CHAT_MODEL.format(user_prompt=question)},
    ]
    fc_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    return fc_prompt

class MCPSmolMind:
    def __init__(self, model_name=MODEL_NAME, debug=False):
        self.model_name = model_name
        self.debug = debug
        self.functions = []
        self.session = None
        self.exit_stack = None
        
        print(f"Initializing model on device: {DEVICE}")
        self.model = outlines.models.transformers(
            model_name,
            device=DEVICE,
            model_kwargs={
                "trust_remote_code": True,
                "torch_dtype": T_TYPE,
            }
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    async def connect_to_server(self, server_script_path):
        """Conectar a un servidor MCP y obtener las herramientas disponibles.

        Args:
            server_script_path: Ruta al script del servidor (.py o .js)
        """
        from contextlib import AsyncExitStack
        
        self.exit_stack = AsyncExitStack()
        
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        print(f"Connecting to MCP server: {server_script_path}")
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # Listar herramientas disponibles
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])
        
        # Convertir herramientas MCP al formato de funciones esperado por SmolMind
        self.functions = []
        for tool in tools:
            func = {
                "name": tool.name,
                "description": tool.description or f"Function {tool.name}",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            
            # Convertir el esquema de entrada a propiedades de función
            if hasattr(tool, "inputSchema") and tool.inputSchema:
                if isinstance(tool.inputSchema, dict):
                    # Extraer propiedades
                    properties = tool.inputSchema.get("properties", {})
                    func["parameters"]["properties"] = properties
                    
                    # Extraer parámetros requeridos
                    required = tool.inputSchema.get("required", [])
                    func["parameters"]["required"] = required
                
            self.functions.append(func)
        
        if not self.functions:
            print("Warning: No functions found from MCP server")

    async def process_query(self, user_prompt):
        """Procesar una consulta de usuario usando SmolMind y las herramientas MCP."""
        if not self.functions:
            return "No functions available. Please connect to an MCP server first."
        
        try:
            # Generar la llamada a función mediante el modelo
            prompt = instruct_prompt(user_prompt, self.functions, self.tokenizer)
            
            # Obtener respuesta del modelo (sin usar regex)
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
            output = self.model.generate(input_ids, max_new_tokens=50, do_sample=False)
            response = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            if self.debug:
                print(f"Functions: {self.functions}")
                print(f"Prompt: {prompt}")
                print(f"Generated response: {response}")
            
            # Extraer nombre de función y argumentos con regex básico
            match = re.match(r'\[(.*?)\((.*?)\)\]', response)
            if not match:
                return f"Could not parse function call: {response}"
            
            function_name = match.group(1)
            args_str = match.group(2)
            
            # Convertir argumentos a diccionario
            args_dict = {}
            if args_str:
                # Expresión regular para extraer pares clave-valor
                pattern = r'(\w+)=("[^"]*"|\'[^\']*\'|\d+|\w+)'
                for key, value in re.findall(pattern, args_str):
                    # Limpiar valores string
                    if value.startswith('"') or value.startswith("'"):
                        value = value[1:-1]
                    # Convertir valores numéricos
                    elif value.isdigit():
                        value = int(value)
                    elif value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    args_dict[key] = value
            
            # Ejecutar la llamada a la herramienta MCP
            print(f"\nCalling MCP tool: {function_name} with args: {args_dict}")
            result = await self.session.call_tool(function_name, args_dict)
            
            return f"\nResponse from {function_name}:\n{result.content}"
        
        except Exception as e:
            return f"Error processing query: {str(e)}"

    async def chat_loop(self):
        """Ejecutar un bucle de chat interactivo."""
        print("\nMCP SmolMind Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print(response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Limpiar recursos."""
        if self.exit_stack:
            await self.exit_stack.aclose()

async def main():
    parser = argparse.ArgumentParser(description="MCP SmolMind Client")
    parser.add_argument("server_path", help="Path to the MCP server script")
    parser.add_argument("--model", default=MODEL_NAME, help="Name or path to the model")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    client = MCPSmolMind(model_name=args.model, debug=args.debug)
    
    try:
        await client.connect_to_server(args.server_path)
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())