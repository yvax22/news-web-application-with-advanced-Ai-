using System.ComponentModel.DataAnnotations.Schema;

[Table("ChatHistory")]
public class ChatHistory
{
    public int Id { get; set; }
    public string UserId { get; set; } = null!;
    public string Question { get; set; } = null!;
    public string Answer { get; set; } = null!;
    public DateTime CreatedAt { get; set; }
}