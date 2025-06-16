using System.Collections.Generic;
using FND.Models;
namespace FND.ViewModel
{
    public class ChatViewModel
    {
        public string Question { get; set; }
        public List<ChatMessage> Messages { get; set; } = new();
    }
}
