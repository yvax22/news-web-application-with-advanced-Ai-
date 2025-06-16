using FND.Models;

namespace FND.ViewModels
{
    public class NewsWithCommentsViewModel
    {
        public News News { get; set; } // خبر واحد
        public List<Comment> Comments { get; set; } = new List<Comment>(); // التعليقات المرتبطة بالخبر
        public string NewCommentContent { get; set; } // محتوى التعليق الجديد
    }
}
