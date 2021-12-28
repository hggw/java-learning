import lombok.Data;
import org.hibernate.validator.constraints.Length;

import java.util.List;

@Data
public class Son {

    List<@Length(max=10,message = "超长") String> id;

}
