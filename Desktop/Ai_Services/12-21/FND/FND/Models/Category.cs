using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace FND.Models
{
    public class Category
    {
        [Key]
        [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
        public int Id { get; set; } // Primary Key
        public string Name { get; set; } = string.Empty;

       

        // Navigation property
        public ICollection<News> News { get; set; } = new List<News>();
    }
}