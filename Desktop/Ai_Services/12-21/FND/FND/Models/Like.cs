using FND.Models;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
namespace FND.Models
{
    

    public class Like
    {
        [Key]
        public int Id { get; set; } 

        public string? UserId { get; set; }
        public int NewsId { get; set; }
        public DateTime CreatedDate { get; set; }
        [ForeignKey("UserId")]

        public virtual ApplicationUser User { get; set; } = new ApplicationUser();
        [ForeignKey("NewsId")]

        public virtual News News { get; set; } 
    }
}
